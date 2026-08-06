import torch
import time
from typing import Callable
import numpy as np

MIN_DIGITS = {torch.float64: 10.0, torch.float32: 5.0}

class OptiMix():
    """
    L-BFGS as the main driver, Adam as a fallback
    """

    def __init__(
        self,
        params: list,
        compute_loss: Callable,
        adam_lr = 0.01,
        lbfgs_lr = 1,
        lbfgs_line_search_fn = "strong_wolfe",
        lbfgs_max_iter = 20,
        lbfgs_history_size = 20,
    ):
        self.params = params
        self.compute_loss = compute_loss
        self.adam_lr              = adam_lr
        self.lbfgs_lr             = lbfgs_lr
        self.lbfgs_line_search_fn = lbfgs_line_search_fn
        self.lbfgs_max_iter       = lbfgs_max_iter
        self.lbfgs_history_size   = lbfgs_history_size
        self.adam_step = 0
        self.adam_total = 0
        self.set_adam()
        self.set_lbfgs()

    def closure(self):

        self.Adam.zero_grad()

        (
            loss,
            self.logdet_V,
            self.quad,
            self.k_reml,
            self.const,
            self.L
        ) = self.compute_loss(return_everything = True)

        self.loglik = loss.item()

        loss.backward()
        return loss

    def run(
        self,
        n_epoch: int = 10_000,
    ):
        self.loss = []
        self.converged = False
        self.previous = torch.inf
        self.tol      = torch.inf

        self.start = time.time()

        for _ in range(n_epoch):
            self.step()
            if self.converged:
                break

        self.duration = time.time() - self.start

    def set_adam(self):
        self.Adam = torch.optim.Adam(
            self.params,
            lr = self.adam_lr,
        )

    def set_lbfgs(self):
        self.LBFGS = torch.optim.LBFGS(
            self.params,
            lr             = self.lbfgs_lr,
            max_iter       = self.lbfgs_max_iter,
            history_size   = self.lbfgs_history_size,
            line_search_fn = self.lbfgs_line_search_fn,
        )

    def step(self):
        self.snap = [p.detach().clone() for p in self.params]

        try:
            self.LBFGS.step(self.closure)
            current = self.loglik
            self.loss.append(current)
            self.set_adam()
            self.adam_step = 0

            if abs(current - self.previous) < self.tolerance():
                self.converged = True

        except RuntimeError:
            with torch.no_grad():
                for param, param_snap in zip(self.params, self.snap):
                    param.copy_(param_snap)
            
            self.set_lbfgs()
            self.adam_step += 1
            self.adam_total += 1
            self.closure()
            self.Adam.step()
            self.loss.append(self.loglik)

        self.previous = self.loglik

    def tolerance(self):
        """
        Adaptive relative tolerance on -2logL, sized on the number of significant
        digits that survive the numerical evaluation of the likelihood.

        A hard-coded absolute threshold silently demands more significant digits
        as the sample size grows, since -2logL is O(n). Budgeting digits instead
        makes the stopping criterion scale-free.

        The budget is machine_digits minus three independent sources of precision
        loss. Every one of them is a LOWER bound on the true loss, so the retained
        digit count is an upper bound on what is actually available: the criterion
        can only ever be too strict (a few extra iterations), never too loose
        (degraded variance components).
        """
        
        terms = [self.logdet_V, self.quad, self.k_reml, self.const]
        dtype = self.L.dtype

        # hard ceiling of the storage format: 15.65 (f64) / 6.92 (f32)
        machine_digits = -np.log10(torch.finfo(dtype).eps)

        # cancellation: each addend carries an absolute error |t|*eps and those
        # add up, inflating the relative error of the sum by sum|t| / |sum t|.
        # Exact, and >= 0 by the triangle inequality.
        cancellation = np.log10(sum(abs(t) for t in terms) / abs(self.loglik))

        # roundoff accumulated over ~m^3/3 flops. Random-walk growth sqrt(m),
        # not Wilkinson's worst case m (aligned signs, pessimistic by 1-2 orders).
        roundoff = 0.5 * np.log10(self.L.shape[0])

        # conditioning: forward error ~ growth * kappa * eps. Dominates when a
        # variance component drifts to zero. kappa_2(M) = kappa_2(L)**2,
        # approximated by the ratio of the extreme diagonal entries of L: exact
        # for diagonally dominant M, collapses to 1 when off-diagonal entries
        # dominate -> understates, never overstates.
        diag = torch.diagonal(self.L).abs()
        kappa = (diag.max() / diag.min().clamp_min(torch.finfo(dtype).tiny)) ** 2
        conditioning = np.log10(float(kappa))

        # the three terms bound the precision AVAILABLE, not the precision
        # REQUIRED for the variance components to be settled. Floor the retained
        # digits so a badly conditioned model cannot buy an arbitrarily loose
        # stopping criterion.
        retained = max(
            machine_digits - (cancellation + roundoff + conditioning),
            MIN_DIGITS[dtype],
        )

        # scaled on the current loglik only: an early outlier must not widen it
        self.tol = abs(self.loglik) * 10.0 ** -retained
        return self.tol