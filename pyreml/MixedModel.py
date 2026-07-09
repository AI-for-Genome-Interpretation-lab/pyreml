import types
from typing import Callable

import numpy as np
from scipy.linalg import block_diag
import pandas as pd
import patsy
import torch
import torch.nn as nn
import math
import time

from .Optimizer import OptiMix
from .GaussianComponents import Random, Residual

class MixedModel:

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        response: str | list[str],
        fixed: str = "1",
        random: None | Random | list[Random] = None,
        residual: Residual | None = None,
        SMW: bool | None = None,
        device: str = "cpu",
    ):

        response = [response] if isinstance(response, str) else list(response)
        residual = Residual() if residual is None else residual
        random = [] if random is None else (random if isinstance(random, list) else [random])

        ## Filter NAs
        missing = [
            c
            for r in random
            for c in ([r.unit] if isinstance(r.unit, str) else list(r.unit))
            if c not in data.columns
        ]
        if missing:
            raise ValueError(f"random grouping variables not found in data: {missing}")
        
        keep = data.index
        for formula in [fixed] + [r.formula for r in random]:
            usable = patsy.dmatrix(
                formula, data=data, NA_action="drop", return_type="dataframe"
            ).index
            keep = keep[keep.isin(usable)]

            any_response = data[response].notna().any(axis=1)
            keep = keep[keep.isin(data.index[any_response])]

        unit_cols = [
            c
            for r in random
            for c in ([r.unit] if isinstance(r.unit, str) else list(r.unit))
        ]
        if unit_cols:
            keep = keep[keep.isin(data.dropna(subset=unit_cols).index)]

        data = data.loc[keep].copy()

        masks = [data[resp].notna().to_numpy() for resp in response]

        y = np.hstack(
            [data.loc[m, resp].to_numpy() for resp, m in zip(response, masks)]
        )
        scale = np.array([float(data[resp].std()) for resp in response], dtype=float)
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)

        ## Build X
        dm = patsy.dmatrix(fixed, data=data, return_type="dataframe")
        fixed_names = dm.design_info.column_names
        X_base = np.asarray(dm)
        
        empty = np.where(~X_base.any(axis=0))[0]
        keep = [i for i in range(X_base.shape[1]) if i not in set(empty)]
        X_base = X_base[:, keep]
        fixed_names = [fixed_names[i] for i in keep]

        X = block_diag(*[X_base[m] for m in masks])

        ## build Z and everything random related
        Z_blocks = []
        random_blocks = []
        random_blocks_inv = []
        varparams = []

        for r in random:
            Z_base = r.design(data, response, scale=scale, device = device)
            Z_e = block_diag(*[Z_base[m] for m in masks])

            Z_blocks.append(Z_e)
            random_blocks.append(r.varmeth())
            random_blocks_inv.append(r.varmeth_inv())
            varparams.extend(r.varparams)

        if Z_blocks:
            Z = np.hstack(Z_blocks)
        else:
            Z = None

        W_blocks = residual.design(data, response, scale=scale, device = device)
        W = block_diag(*[W_blocks[m] for m in masks])
        residual.check_Rtrick(W)

        varparams.extend(residual.varparams)

        X = torch.as_tensor(X, dtype=torch.double, device=device)
        Z = torch.as_tensor(Z, dtype=torch.double, device=device) if Z is not None else None
        W = torch.as_tensor(W, dtype=torch.double, device=device)
        y = torch.as_tensor(y, dtype=torch.double, device=device).reshape(-1, 1)

        do_REML = (
            Z is not None
            or len(response) > 1
            or residual.right_hand != "iid"
        )

        def varmeth(self):
            R_tot = residual.varmeth()
            R = self.W @ R_tot() @ self.W.T

            if not random_blocks:
                return None, R

            G = torch.block_diag(*[block() for block in random_blocks])
            return G, R

        def varmeth_inv(self):
            Rinv, logdet_R = residual.varmeth_inv()(self.W)

            if not random_blocks_inv:
                return None, Rinv, None, logdet_R

            inv_logdets = [blk() for blk in random_blocks_inv]
            Ginv = torch.block_diag(*[gi for gi, _ in inv_logdets])
            logdet_G = sum(ld for _, ld in inv_logdets)
            return Ginv, Rinv, logdet_G, logdet_R

        mm = cls(
            y=y,
            X=X,
            Z=Z,
            W=W,
            varmeth = varmeth,
            varmeth_inv = varmeth_inv,
            varparams=varparams,
            do_REML=do_REML,
            device = device,
        )
        mm.response = response
        mm.fixed_names = fixed_names
        mm.residual = residual
        mm.random = [rand for rand in random]

        if not mm.residual.Rtrick :
            mm.SMW = False
            
        if SMW is not None:
            mm.SMW = SMW

        return mm

    def __init__(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        Z: None | torch.Tensor,
        W: None | torch.Tensor,
        varparams: list[dict],
        varmeth: Callable | None = None,
        varmeth_inv: Callable | None = None,
        do_REML: bool = True,
        device = "cpu",
    ):
        self.device = device

        if W is None:
            W = torch.eye(len(y))
        self.y = y
        self.X = X
        self.Z = Z
        self.W = W
        self.n, self.p = X.shape
        self.q = Z.shape[1] if Z is not None else 0

        if varmeth is None and varmeth_inv is None:
            raise ValueError("At least one of varmeth or varmeth_inv must be provided")
        elif varmeth_inv is None:            # seule la voie directe existe
            self.SMW = False
        elif varmeth is None:                # seule la voie Woodbury existe
            self.SMW = True
        else:                                # les deux : on choisit par dimension
            self.SMW = (self.q < self.n)

        self.beta = nn.Parameter(torch.zeros(self.X.shape[1], 1, dtype=torch.double, device = device))

        self.varparams   = varparams
        self.do_REML     = do_REML
        self.varmeth     = types.MethodType(varmeth, self) if varmeth is not None else None
        self.varmeth_inv = types.MethodType(varmeth_inv, self) if varmeth_inv is not None else None

        self.opti_REML = OptiMix(
            params=[self.beta, *(p["tensor"] for p in self.varparams)],
            closure=self.REML_closure,
        )
        if Z is not None:
            self.uhat = nn.Parameter(torch.zeros(self.Z.shape[1], 1, dtype=torch.double, device = device))

    def migrate(self, dtype: torch.dtype):
        pass

    def log(self):
        if not self._log:
            print("(no log)")
            return

        head = self._log[0]
        entries = self._log[1:]

        def fmt(key, v):
            if isinstance(v, bool):
                return str(v)
            if "loss" in key and isinstance(v, (int, float)):
                return f"{float(v):.10f}"
            if isinstance(v, float):
                return f"{v:.6g}"
            if v is None:
                return ""
            return str(v)

        # ---- blocs (titre, [(clé, valeur), ...]) ----
        blocks = []

        # platform
        plat = [("device", str(head.get("device", "")))]
        if "gpu" in head:
            plat.append(("gpu", str(head["gpu"])))
        if "threads" in head:
            plat.append(("n threads", str(head["threads"])))
        plat.append(("torch", str(head.get("torch", ""))))
        if "cuda" in head:
            plat.append(("cuda", str(head["cuda"])))
        blocks.append(("platform", plat))

        # model
        model_keys = ["n obs", "n fixed effects", "n random effects",
                      "SMW", "variance parameters"]
        model = [(k, fmt(k, head[k])) for k in model_keys if k in head]
        blocks.append(("model", model))

        # steps
        for e in entries:
            step = e.get("step", "")
            dtype = e.get("dtype", "")
            title = f"{step} · {dtype}" if dtype else step
            kv = [(k, fmt(k, v)) for k, v in e.items()
                  if k not in ("step", "dtype")]
            blocks.append((title, kv))

        # ---- largeurs ----
        w_k = max(len(k) for _, kv in blocks for k, _ in kv)
        w_v = max(len(v) for _, kv in blocks for _, v in kv)
        content_w = 4 + w_k + 2 + w_v
        titles_w = max(len(t) for t, _ in blocks) + 2
        inner = max(content_w, titles_w, 58)      # 58 -> boîte de 60 avec bordures

        def top():    return "╭" + "─" * (inner + 2) + "╮"
        def bottom(): return "╰" + "─" * (inner + 2) + "╯"
        def sep():    return "├" + "─" * (inner + 2) + "┤"
        def pad(s):   return "│ " + s + " " * (inner - len(s)) + " │"

        # ---- timestamp collé à gauche ----
        t0 = head.get("t0")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)) if t0 else ""
        print(stamp)

        print(top())
        for i, (title, kv) in enumerate(blocks):
            if i > 0:
                print(sep())
            print(pad("  " + title))
            print(pad("  " + "─" * len(title)))
            for k, v in kv:
                print(pad("    " + k.ljust(w_k) + "  " + v))
        print(bottom())

    def fit(self):

        t0 = time.time()
        info = {
            "device": self.device,
            "t0": t0,
            "n obs": self.n,
            "n fixed effects": self.p,
            "n random effects": self.q,
            "SMW": self.SMW,
            "torch": torch.__version__,
        }
        if self.device == "cpu":
            info["threads"] = torch.get_num_threads()
        elif self.device.startswith("cuda"):
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(self.device)

        _log = []

        if self.do_REML:

            self.migrate(torch.float)
            self.OLS(terminate = False)
            t1 = time.time()
            _log.append({
                "step": "OLS",
                "dtype": "float",
                "time": t1 - t0,
            })

            self.REML(convergence = 1e-5)
            t2 = time.time()
            _log.append({
                "step": "REML",
                "dtype": "float",
                "time": t2 - t1,
                "convergence": self.opti_REML.converged,
                "n steps total": len(self.opti_REML.loss),
                "n steps adam": self.opti_REML.adam_total,
                "REML loss": self.opti_REML.loss[-1],
            })

            self.migrate(torch.double)
            self.REML(convergence = 1e-10)
            t3 = time.time()
            _log.append({
                "step": "REML",
                "dtype": "double",
                "time": t3 - t2,
                "convergence": self.opti_REML.converged,
                "n steps total": len(self.opti_REML.loss),
                "n steps adam": self.opti_REML.adam_total,
                "REML loss": self.opti_REML.loss[-1],
            })
            
            self.HMME()
            _log.append({
                "step": "HMME",
                "dtype": "double",
                "time": time.time() - t3,
                "REML loss": self.neg2loglik,
            })
        
        else:
            self.migrate(torch.double)
            self.OLS(terminate = True)
            _log.append({
                "step": "OLS",
                "dtype": "double",
                "time": time.time() - t0,
                "ML loss": self.neg2loglik,
            })

        info["variance parameters"] = self.df_var
        self._log = [info] + _log

        return self

    def OLS(
        self,
        terminate: bool = False,
    ):
        """
        Estimate fixed effects with ordinary least squares, beta = (X'X)^-1 X'y
        """
        with torch.no_grad():

            XtX = self.X.T @ self.X
            Xty = self.X.T @ self.y

            b = torch.linalg.solve(XtX, Xty)

            resid = self.y - self.X @ b
            sigma2 = (resid.T @ resid).squeeze() / (self.n - self.p)
            EEV = sigma2 * torch.linalg.inv(XtX)

        self.beta.data.copy_(b)

        if terminate:
            sd = self.residual.scale_d()
            if sd is None:
                log_s0 = torch.log(sigma2)
            else:
                log_s0 = torch.log(sigma2) - 2.0 * torch.log(sd.reshape(()))
            
            self.residual.log_S.data.copy_(log_s0)
            self.residual.format_variance()

            self.EEV = EEV
            self.format_fixed()

            residuals = (self.y - self.X @ self.beta).flatten()
            self.residual.format_residuals(residuals, self.W)

            self.compute_AIC(REML = False)
    
    def ML_loss(self):

        if self.residual.log_S.numel() != 1:
            raise ValueError(
                f"ML_loss expects a scalar residual variance (iid OLS path), "
                f"got {self.residual.log_S.numel()} elements."
            )
        
        s2  = self.residual.build_S().reshape(())
        s2_ML = s2 * (self.n - self.p) / self.n
        const = self.n * math.log(2*math.pi)
        logdet_V = self.n * torch.log(s2_ML)
        quad = self.n
        return logdet_V + quad + const
    
    def REML(
        self,
        n_epoch: int = 10_000,
        convergence: float = 1e-10,
    ):
        """
        Restricted maximum likelihood estimation of the variance components + beta
        - n_epochs: the number of epochs
        - convergence; the convergence criterion.
        """

        self.opti_REML.run(
            n_epoch=n_epoch,
            convergence=convergence,
        )

        for rand in getattr(self, "random", []):
            rand.format_variance()

        residual = getattr(self, "residual", None)
        if residual is not None:
            residual.format_variance()

    def REML_closure(self):

        self.opti_REML.Adam.zero_grad()

        loss = self.REML_loss()
        loss.backward()
        return loss

    def REML_loss(self):

        r = self.y - self.X @ self.beta
        const    = (self.n - self.p) * math.log(2 * math.pi)

        if self.SMW:
            # ---- Sherman–Morrison–Woodbury: from structured inverses, V never formed ----
            Ginv, Rinv, logdet_G, logdet_R = self.varmeth_inv()

            if self.Z is None:
                logdet_V = logdet_R
                quad     = (r.T @ Rinv @ r).squeeze()
                k_reml   = torch.logdet(self.X.T @ Rinv @ self.X)
            
            else:
                Z = self.Z
                P  = Ginv + Z.T @ Rinv @ Z                        # (q, q) capacitance
                Lp = torch.linalg.cholesky(P)
                logdet_P = 2.0 * torch.sum(torch.log(torch.diagonal(Lp)))
                logdet_V = logdet_R + logdet_G + logdet_P         # determinant lemma

                # r' V^-1 r = r'R⁻¹r − (Z'R⁻¹r)' P⁻¹ (Z'R⁻¹r)
                Rir   = Rinv @ r
                ZtRir = Z.T @ Rir
                quad  = (r.T @ Rir).squeeze() \
                    - (ZtRir.T @ torch.cholesky_solve(ZtRir, Lp)).squeeze()

                # X' V^-1 X = X'R⁻¹X − (Z'R⁻¹X)' P⁻¹ (Z'R⁻¹X)
                RiX   = Rinv @ self.X
                ZtRiX = Z.T @ RiX
                XtViX = self.X.T @ RiX - ZtRiX.T @ torch.cholesky_solve(ZtRiX, Lp)
                k_reml = torch.logdet(XtViX)

        else:
            # ---- Direct: single Cholesky of V = ZGZ' + R ----
            G, R = self.varmeth()

            V = R if self.Z is None else self.Z @ G @ self.Z.T + R

            Lv = torch.linalg.cholesky(V)
            M  = torch.linalg.solve_triangular(Lv, r, upper=False)

            logdet_V = 2.0 * torch.sum(torch.log(torch.diag(Lv)))
            quad     = (M.T @ M).squeeze()
            k_reml   = torch.logdet(self.X.T @ torch.cholesky_solve(self.X, Lv))
        
        return logdet_V + quad + k_reml + const

    def HMME(self):
        """
        Henderson's mixed model equations: BLUE of beta, BLUP of u, and the
        associated prediction error variances (PEV). LH is factored once; its
        inverse gives Var(beta_hat) (top-left block) and the PEV of u_hat
        (bottom-right block). If Random/Residual objects are attached (high-level
        build), each effect's slice is dispatched to them; otherwise (low-level
        constructor) the raw results are stored flat on the model.
        """

        with torch.no_grad():
            if callable(self.varmeth_inv):
                Ginv, Rinv, _, _ = self.varmeth_inv()
            else:
                G, R = self.varmeth()
                Ginv = torch.linalg.inv(G)
                Rinv = torch.linalg.inv(R)

            if self.Z is None:
                LH = self.X.T @ Rinv @ self.X
                RH = self.X.T @ Rinv @ self.y
            else:
                XtRiX = self.X.T @ Rinv @ self.X
                XtRiZ = self.X.T @ Rinv @ self.Z
                ZtRiZ = self.Z.T @ Rinv @ self.Z

                LH = torch.cat([
                        torch.cat([XtRiX, XtRiZ], dim=1),
                        torch.cat([XtRiZ.T, ZtRiZ + Ginv], dim=1),
                    ],
                    dim=0,
                )
                RH = torch.cat([
                        self.X.T @ Rinv @ self.y,
                        self.Z.T @ Rinv @ self.y
                    ],
                    dim=0
                )
            
            # Factor LH once (symmetric PD): reuse for the solve and the inverse.
            LH = 0.5 * (LH + LH.T)
            Lchol = torch.linalg.cholesky(LH)
                
            sol = torch.cholesky_solve(RH, Lchol)
            C = torch.cholesky_inverse(Lchol)          # C = LH^{-1}
            
            p = self.p
            self.beta.data.copy_(sol[:p])
            self.EEV = C[:p, :p]

            if self.Z is not None:
                self.uhat.data.copy_(sol[p:])
                PEV = C[p:, p:]

            # Fixed effects: labelled table if high-level, raw beta + EEV otherwise.
            self.format_fixed()

            if self.Z is None:
                y_hat = self.X @ self.beta
            else:
                y_hat = self.X @ self.beta + self.Z @ self.uhat

            residuals = (self.y - y_hat).flatten()

            random = getattr(self, "random", None)
            residual = getattr(self, "residual", None)

            if random is not None:
                offset = 0
                for rand in random:
                    size = rand.k * rand.c * rand.L
                    idx = slice(offset, offset + size)
                    rand.format_pred(self.uhat[idx], PEV[idx, idx])
                    offset += size
            else:
                self.PEV = PEV
                
            if residual is not None:
                residual.format_residuals(residuals, self.W)
            else:
                self.residuals = residuals

            self.compute_AIC(REML = True)

    def format_fixed(self):
        """
        Format the fixed-effect estimates.

        High-level build (fixed_names + response stored on the model): a long
        DataFrame in self.estimates, kept in the native order of beta
        (response-outer, term-inner) so that row i aligns with row/col i of the
        estimation error variance matrix self.EEV. Columns are
        [response | term | estimate | SE = sqrt(diag(EEV)) | t = estimate / SE ]

        Low-level fallback: self.estimates is not built; the raw beta vector
        (self.beta) and self.EEV are the only outputs.
        """
        fixed_names = getattr(self, "fixed_names", None)
        response = getattr(self, "response", None)
        if fixed_names is None or response is None:
            return  # fallback: raw beta + EEV only

        beta = self.beta.detach().flatten().tolist()
        p_fixed = len(fixed_names)

        se = torch.sqrt(torch.diag(self.EEV)).detach().flatten().tolist()
        rows = [
            (
                resp,
                term,
                beta[r * p_fixed + t],
                se[r * p_fixed + t],
                beta[r * p_fixed + t] / se[r * p_fixed + t],
            )
            for r, resp in enumerate(response)
            for t, term in enumerate(fixed_names)
        ]
        self.estimates = pd.DataFrame(
            rows, columns=["response", "term", "estimate", "SE", "t"]
        )

    def compute_AIC(self, REML = True):
        """
        -2logL_REML at convergence + parameter counts -> AIC.
        not designed for the low level constructor (the number
        of independent parameters cannot be automatically computed)
        """

        if getattr(self, "residual", None) is None:
            return
        
        self.df_beta = len(self.beta)
        randoms = getattr(self, "random", [])
        residual = getattr(self, "residual", None)
        self.df_var = sum(c.n_params for c in randoms)
        self.df_var += residual.n_params
        self.n_params = self.df_beta + self.df_var
        
        with torch.no_grad():
            if REML:
                self.neg2loglik = float(self.REML_loss().detach())
                self.AIC_meth = "REML"
            else:
                self.neg2loglik = float(self.ML_loss().detach())
                self.AIC_meth = "ML"

        self.AIC = self.neg2loglik + 2 * self.n_params

