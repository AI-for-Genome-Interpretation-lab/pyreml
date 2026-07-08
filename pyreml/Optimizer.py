import torch
from importlib.resources import files
import time
from typing import Callable

class Optimizer:

    def __init__(
        self,
        params: list,
        closure: Callable,
    ):
        self.params = params
        self.closure = closure

    def run(
        self,
        n_epoch: int = 10_000,
        convergence: float = 1e-10,
    ):
        self.loss = []
        self.converged = False
        self.criterion = convergence

        self.start = time.time()

        for _ in range(n_epoch):
            self.step()
            if self.converged:
                break

        self.duration = time.time() - self.start

class OptiMix(Optimizer):
    """
    L-BFGS as the main driver, Adam as a fallback
    """

    def __init__(
        self,
        params: list,
        closure: Callable,
        adam_lr = 0.01,
        lbfgs_lr = 1,
        lbfgs_line_search_fn = "strong_wolfe",
        lbfgs_max_iter = 20,
        lbfgs_history_size = 20,
    ):
        super().__init__(params, closure)

        self.monitor_idx = list(range(len(params)))

        self.adam_lr              = adam_lr
        self.lbfgs_lr             = lbfgs_lr
        self.lbfgs_line_search_fn = lbfgs_line_search_fn
        self.lbfgs_max_iter       = lbfgs_max_iter
        self.lbfgs_history_size   = lbfgs_history_size
        self.adam_step = 0
        self.adam_total = 0
        self.set_adam()
        self.set_lbfgs()

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
            loss = self.LBFGS.step(self.closure)
            self.loss.append(loss.item())
            self.set_adam()
            self.adam_step = 0

            self.delta = max(
                (self.params[i].detach() - self.snap[i]).abs().max().item()
                for i in self.monitor_idx
            )
            if self.delta < self.criterion:
                self.converged = True

        except RuntimeError:
            with torch.no_grad():
                for param, param_snap in zip(self.params, self.snap):
                    param.copy_(param_snap)
            
            self.set_lbfgs()
            self.adam_step += 1
            self.adam_total += 1
            loss = self.closure()
            self.Adam.step()
            self.loss.append(loss.item())
