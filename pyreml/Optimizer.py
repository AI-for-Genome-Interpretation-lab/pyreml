import torch
import time
from typing import Callable
import numpy as np

MAX_DIGITS = 10.0
MAX_ABS_TOL = 0.02 # 1/100 for a difference of 2 points of -2loglik

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
            logdet_V,
            quad,
            k_reml,
            const,
            L
        ) = self.compute_loss(return_everything = True)

        # detach: these feed the tolerance computation only, never the graph
        self.logdet_V = float(logdet_V.detach())
        self.quad     = float(quad.detach())
        self.k_reml   = float(k_reml.detach())
        self.const    = float(const)
        self.L        = L.detach()
        self._workingloss = loss.item()

        loss.backward()
        return loss

    def run(
        self,
        n_epoch: int = 10_000,
        max_digits = None,
    ):
        self.loss = []
        self.converged = False
        self.degenerate = False
        self.previous = torch.inf
        self.digits_tolerance_total = torch.inf
        self.digits_machine         = torch.inf
        self.digits_cancellation    = torch.inf
        self.digits_roundoff        = torch.inf
        self.digits_conditionning   = torch.inf

        self.start = time.time()

        for _ in range(n_epoch):
            self.step(max_digits=max_digits)
            if self.converged:
                if self.degenerate:
                    self.converged = False
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

    def step(self, max_digits = None):
        self.snap = [p.detach().clone() for p in self.params]

        try:
            loss = self.LBFGS.step(self.closure)
            current = loss.item()
            self.loss.append(current)
            self.set_adam()
            self.adam_step = 0

            relative_tol = self.tolerance(max_digits)
            absolute_tol = abs(current) * 10.0 ** -relative_tol

            # degenerate case: back to the hard-coded absolute threshold
            # => maybe it's a hard beginning, maybe it's degenerate and we
            # seek stagnation
            if absolute_tol > MAX_ABS_TOL:
                self.degenerate = True
                absolute_tol = MAX_ABS_TOL
            else:
                self.degenerate = False

            self.absolute_tolerance = absolute_tol
            
            if abs(current - self.previous) < absolute_tol:
                self.converged = True

        except RuntimeError:
            with torch.no_grad():
                for param, param_snap in zip(self.params, self.snap):
                    param.copy_(param_snap)
            
            self.set_lbfgs()
            self.adam_step += 1
            self.adam_total += 1
            loss = self.closure()
            current = loss.item()
            self.Adam.step()
            self.loss.append(current)

        self.previous = current

    def tolerance(self, max_digits = None):
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
        self.digits_machine = -np.log10(torch.finfo(dtype).eps)

        # cancellation: each addend carries an absolute error |t|*eps and those
        # add up, inflating the relative error of the sum by sum|t| / |sum t|.
        # Exact, and >= 0 by the triangle inequality.
        cancellation = sum(abs(t) for t in terms) / abs(self._workingloss)
        self.digits_cancellation = np.log10(cancellation)

        # roundoff accumulated over ~m^3/3 flops. Random-walk growth sqrt(m),
        # not Wilkinson's worst case m (aligned signs, pessimistic by 1-2 orders).
        self.digits_roundoff = np.log10(np.sqrt(self.L.shape[0]))

        # conditioning: forward error ~ growth * kappa * eps. Dominates when a
        # variance component drifts to zero. kappa_2(M) = kappa_2(L)**2,
        # approximated by the ratio of the extreme diagonal entries of L: exact
        # for diagonally dominant M, collapses to 1 when off-diagonal entries
        # dominate -> understates, never overstates.
        diag = torch.diagonal(self.L).abs()
        kappa = (diag.max() / diag.min().clamp_min(torch.finfo(dtype).tiny)) ** 2
        self.digits_conditionning = np.log10(float(kappa))

        # cap the demand: 10 significant digits on -2logL is already ample for
        # the variance components, and asking for more only buys noise. The
        # three terms above are all lower bounds on the precision actually
        # lost, so they can only ever make the criterion stricter than needed;
        # the cap is what guarantees a floor on the speed-up.
        retained = min(
            self.digits_machine - (self.digits_cancellation + self.digits_roundoff + self.digits_conditionning),
            MAX_DIGITS if max_digits is None else max_digits,
        )

        # scaled on the current loglik only: an early outlier must not widen it
        self.digits_tolerance_total = retained
        return self.digits_tolerance_total