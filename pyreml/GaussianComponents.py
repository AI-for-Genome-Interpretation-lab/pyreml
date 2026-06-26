from typing import Literal, Callable
import numpy as np
from scipy.linalg import logm
import pandas as pd
import patsy
import torch
import torch.nn as nn
import warnings

class GaussianComponent:

    def __init__(
        self,
        unit: str | None,
        formula: str = "1",
        left_hand: Literal["iid", "diag", "full", "fa"] = "iid",
        right_hand: Literal["iid", "ar", "str", "het"] = "iid",
        covariance: None | np.ndarray = None,
        distance: None | np.ndarray = None,
        matrix_index: None | list = None,
        het_formula: None | str = None,
        n_axes: None | int = None,
        init: None | float | np.ndarray = None,
        jitter: float = 0,
    ):
        """
        The variance of a random effect is defined as u ~ N(0, S ⊗ K)
 
        S is the left-hand factor: the covariance ACROSS the components of the
        effect. A component is a (response, formula-element) pair, so S has
        dimension (k * c) x (k * c) with k the number of responses and c the number
        of columns of `formula`.
 
        K is the right-hand factor: the covariance ACROSS the levels.
 
        `unit` is the grouping COLUMN of the DataFrame (e.g. "genotype"); its
        distinct values are the levels of the effect.
 
        Supported types for S (`left_hand`), with d = k * c:
            - iid:  S = s² * I_d, a single scalar variance (equal
                    variances, no covariance across components) (default)
            - diag: a diagonal covariance matrix, estimated from the data
            - full: a full covariance matrix, estimated from the data
            - fa: factor-analytic approximation of the full structure


        Supported types for K (`right_hand`):
            - iid:  identity matrix, independent levels (default)
            - ar:   exponential autoregressive kernel, K = exp(-ar * D),
                    with ar = exp(log_ar) > 0 estimated and D a known distance
                    matrix between levels (see `distance`)
            - str:  known structure matrix, e.g. a kinship (see `covariance`)
            - het:  diagonal, one variance per column resulting of "het_formula".
                    "het" is primarily meant to be paired with left_hand="iid".
 
        `covariance` is the known structure matrix for right_hand="str".
        `distance`   is the known distance matrix for right_hand="ar".
        `matrix_index` lists the levels of `unit` in the order in which they
            appear in the rows/cols of `covariance` / `distance`. Required
            whenever such a matrix is given for a grouped random effect; not
            allowed for residual structures (`unit=None`).
        `het_formula` is the formula for right_hand="het".
        `n_axes` is the number of factorial axes for left_hand="fa".
        `init` is the initial value of S. For left_hand in {full, diag, fa} it is
            a (k*c) x (k*c) array (or its diagonal for diag); for left_hand="iid"
            it is a scalar. When None, S starts at the identity.
        """

        self.unit = unit
        self.formula = formula
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.covariance = covariance
        self.distance = distance
        self.matrix_index = matrix_index
        self.het_formula = het_formula
        self.n_axes = n_axes
        self.init = init
        self.jitter = jitter

        # check "ar"
        if self.right_hand == "ar":
            if self.distance is None:
                raise ValueError(
                    "`distance` is required when right_hand = 'ar'."
                )
        elif self.distance is not None:
            raise ValueError(
                "`distance` must be None when right_hand != 'ar'."
            )
        
        # check str
        if self.right_hand == "str":
            if self.covariance is None:
                raise ValueError(
                    "`covariance` is required when right_hand = 'str'."
                )
        elif self.covariance is not None:
            raise ValueError(
                "`covariance` must be None when right_hand != 'str'."
            )

        # check het
        if self.right_hand == "het":
            if self.het_formula is None:
                raise ValueError("`het_formula` is required when `right_hand='het'`.")
        elif self.het_formula is not None:
            raise ValueError(
                "`het_formula` must be None when right_hand != 'het'."
            )
        
        if self.right_hand == "het" and self.left_hand != "iid":
            warnings.warn(
                "Using right_hand='het' with left_hand != 'iid' is an unusual configuration.",
                stacklevel=2,
            )
        
        # check fa
        if self.left_hand == "fa":
            if self.n_axes is None:
                raise ValueError(
                    "`n_axes` is required when left_hand = 'fa'. A practical default is n_axes = 2."
                )
        elif self.n_axes is not None:
            raise ValueError(
                "`n_axes` must be None when left_hand != 'fa'."
            )
 
    def init_varparams(self) -> None:
        """
        Instantiate the trainable variance parameters for this effect.

        Left-hand factor S, stored in its log-parameterization (d = k * c):
            - iid:  S = exp(log_S) * I_d              (log_S a scalar)
            - full: S = matrix_exp(log_S + log_S.T)   (log_S a (d, d) matrix)
            - diag: S = diag(exp(log_S))              (log_S a (d,) vector)
            - fa:   S = Q diag(Lambda) Q.T + diag(Psi), with Q, _ = qr(M).
                    The three blocks (M (d, q), log_Lambda (q,), log_Psi (d,))
                    are concatenated into a single flat parameter `log_S` of
                    length d*q + q + d, in that order; build_S slices it back
                    with the known dimensions.
 
        Right-hand factor K:
            - iid / str: no trainable parameter
            - ar:  a single log_ar (ar = exp(log_ar) > 0), initialized at 0
            - het: p-1 trainable log-ratios log_h (the reference modality is
                   fixed at variance 1). Requires make_V(data) beforehand.
        """
        self.varparams = []

        # components of the effect, response-outer / element-inner: matches S order
        self.components = [
            (resp, name) for resp in self.responses for name in self.colnames
        ]
        d = self.k * self.c

        # --- left-hand factor S ---------------------------------------------
        match self.left_hand:

            case "iid":
                if self.init is None:
                    log_S0 = torch.zeros((), dtype = self.dtype, device = self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    if init.size != 1:
                        raise ValueError("init for left_hand='iid' must be a scalar.")
                    log_S0 = torch.as_tensor(np.log(float(init)), dtype = self.dtype, device = self.device)
                self.log_S = nn.Parameter(log_S0)
                n_left = 1

            case "full":
                if self.init is None:
                    log_S0 = torch.zeros(d, d, dtype = self.dtype, device = self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    log_S0 = torch.as_tensor(np.real(logm(init)) / 2.0, dtype=torch.double, device=self.device)
                self.log_S = nn.Parameter(log_S0)
                n_left = d * (d + 1) // 2

            case "diag":
                if self.init is None:
                    log_S0 = torch.zeros(d, dtype = self.dtype, device = self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    diag = np.diag(init) if init.ndim == 2 else init
                    log_S0 = torch.as_tensor(np.log(diag), dtype = self.dtype, device = self.device)
                self.log_S = nn.Parameter(log_S0)
                n_left = d
            
            case "fa":
                q = self.n_axes
                if not (1 <= q <= d):
                    raise ValueError(
                        f"n_axes must satisfy 1 <= n_axes <= {d} (= k*c) for left_hand='fa', got {q}."
                    )

                # target covariance S0: identity by default, else the user matrix
                if self.init is None:
                    S0 = torch.eye(d, dtype = self.dtype, device = self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    if init.shape != (d, d):
                        raise ValueError(
                            f"init for left_hand='fa' must be a ({d}, {d}) covariance matrix."
                        )
                    S0 = torch.as_tensor(init, dtype = self.dtype, device = self.device)

                # q dominant eigenpairs (eigh returns ascending order)
                eigvals, eigvecs = torch.linalg.eigh(S0)
                Lambda0 = eigvals.flip(0)[:q]          # (q,), descending
                Q0 = eigvecs.flip(1)[:, :q]            # (d, q)

                # diagonal residual carried by Psi (PSD residual => Psi0 >= 0)
                low_rank = (Q0 * Lambda0) @ Q0.T
                Psi0 = torch.diagonal(S0) - torch.diagonal(low_rank)

                log_Lambda0 = torch.log(torch.clamp(Lambda0, min=1e-8))
                log_Psi0    = torch.log(torch.clamp(Psi0   , min=1e-8))

                # M0 = Q0 @ R0, R0 upper-triangular of ones (positive diagonal):
                # qr(M0) recovers Q0 up to column signs, which leave S unchanged.
                R0 = torch.triu(torch.ones(q, q, dtype = self.dtype, device = self.device))
                M0 = Q0 @ R0

                flat0 = torch.cat([M0.reshape(-1), log_Lambda0, log_Psi0])
                self.log_S = nn.Parameter(flat0)
                # loadings (d*q) + specificity Psi (d), 
                # minus the rotational indeterminacy of 
                # the q factors: dim O(q) = q(q-1)/2 directions
                n_left = d * q + d - q * (q - 1) // 2

            case _:
                raise ValueError(f"unsupported left hand type: {self.left_hand}")

        self.varparams.append({
            "effect": self.unit if self.unit is not None else "residual",
            "element": "S",
            "tensor": self.log_S,
            "index": self.components,
            "structure": self.left_hand,
        })

        # --- right-hand factor K --------------------------------------------
        match self.right_hand:

            case "iid":
                n_right = 0

            case "ar":
                if self.distance is None:
                    raise ValueError("a distance matrix `distance` must be provided for right_hand='ar'")
                self.log_ar = nn.Parameter(torch.zeros((), dtype = self.dtype, device = self.device))
                self.varparams.append({
                    "effect": self.unit if self.unit is not None else "residual",
                    "element": "ar",
                    "tensor": self.log_ar,
                    "structure": "ar",
                })
                n_right = 1

            case "str":
                if self.covariance is None:
                    raise ValueError("a covariance matrix must be provided for right_hand='str'")
                # constant, no trainable parameter
                n_right = 0

            case "het":
                if not hasattr(self, "V"):
                    raise ValueError(
                        "make_V(data) must be called before init_varparams for right_hand='het'."
                    )
                self.log_h = nn.Parameter(torch.zeros(self.n_het, dtype = self.dtype, device = self.device))
                self.varparams.append({
                    "effect": self.unit if self.unit is not None else "residual",
                    "element": "het",
                    "tensor": self.log_h,
                    "index": self.het_index,
                    "structure": "het",
                })
                n_right = self.n_het

            case _:

                raise ValueError(f"unsupported right hand type: {self.right_hand}")
            
        self.n_params = n_left + n_right
 
    def make_V(self, data: pd.DataFrame) -> None:
        """
        Build the het incidence V for right_hand="het" from a patsy formula.

        The variance multiplier at the granularity of K is exp(V @ h), with
        V = dmatrix(het_formula). The Intercept column anchors the scale: its
        coefficient is fixed to 0 (multiplier exp(0)=1, carried by the left-hand
        factor S), so only the remaining columns get a trainable log-coefficient.
        The Intercept is located by NAME and moved to column 0, so build_K can
        prepend a single fixed 0 regardless of where patsy placed it.

        Granularity:
            - residual (unit=None): one row per observation,
            - grouped: one row per level of `unit`, taken at the first occurrence
            of each level and aligned to self.index.

        Sets self.V (Intercept first), self.het_index (non-intercept column names),
        self.n_het (p - 1 trainable coefficients).
        """
        if self.het_formula is None:
            raise ValueError("`het_formula` is required for right_hand='het'.")

        if self.unit is None:
            frame = data
        else:
            frame = data.groupby(self.unit, sort=False).first().reindex(self.index)

        V_df = patsy.dmatrix(
            self.het_formula, data=frame,
            return_type="dataframe", NA_action="raise",
        )

        cols = list(V_df.columns)
        if "Intercept" not in cols:
            raise ValueError(
                "an intercept is required in `het_formula`: it anchors the variance "
                "scale carried by the left-hand factor S (remove any '0 +' / '- 1')."
            )

        # locate Intercept by name, move it to column 0
        rest = [c for c in cols if c != "Intercept"]
        V_df = V_df[["Intercept"] + rest]

        self.V = torch.tensor(V_df.to_numpy(), dtype = self.dtype, device = self.device)
        self.het_index = rest
        self.n_het = len(rest)
 
    def build_S(self) -> torch.Tensor:
        """
        Rebuild the left-hand factor S from its log-parameterization.
        Differentiable in self.log_S.
        """
        log_S = self.log_S

        match self.left_hand:

            case "iid":
                d = self.k * self.c
                return torch.exp(log_S) * torch.eye(d, dtype = self.dtype, device = self.device)
            
            case "full":
                return torch.linalg.matrix_exp(log_S + log_S.T)
            
            case "diag":
                return torch.diag(torch.exp(log_S))
            
            case "fa":
                d = self.k * self.c
                q = self.n_axes
                M = log_S[: d * q].reshape(d, q)
                log_Lambda = log_S[d * q : d * q + q]
                log_Psi = log_S[d * q + q :]

                Q, _ = torch.linalg.qr(M)
                Lambda = torch.exp(log_Lambda)
                Psi = torch.exp(log_Psi)
                return (Q * Lambda) @ Q.T + torch.diag(Psi)

            case _:
                raise NotImplementedError(f"left_hand='{self.left_hand}' is not implemented.")

    def build_K(
            self,
            covariance: None | torch.Tensor = None,
            distance: None | torch.Tensor = None,
        ) -> torch.Tensor:
            """
            Build the right-hand factor K. Differentiable when the structure
            carries trainable parameters (ar, het), constant otherwise (iid, str).

            `covariance` / `distance` default to the training matrices stored on the
            effect; passing new ones (over a superset of levels) lets `predict` reuse
            the same kernel logic for kriging.
            """
            covariance = self.covariance if covariance is None else covariance
            distance = self.distance if distance is None else distance

            match self.right_hand:

                case "iid":
                    return torch.eye(self.L, dtype = self.dtype, device = self.device)
                
                case "ar":
                    ar = torch.exp(self.log_ar)
                    return torch.exp(-ar * distance)
                
                case "str":
                    return covariance
                
                case "het":
                    h = torch.cat([torch.zeros(1, dtype = self.dtype, device = self.device), self.log_h])
                    return torch.diag(torch.exp(self.V @ h))
                
                case _:
                    raise ValueError(f"unsupported right hand type: {self.right_hand}")
        
    def varmeth(self) -> Callable:
        """
        Return the unitary function for this effect: a closure that builds this
        effect's block S ⊗ K. It closes over `self`, so it reads the live
        parameters (updated in place by the optimizer) and the stored constants.
        """
        def block() -> torch.Tensor:
            S = self.build_S()
            K = self.build_K()
            abs_jitter = self.jitter * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
            rel_jitter = self.jitter * torch.diag_embed(torch.diagonal(S))
            S = S + abs_jitter + rel_jitter
            return torch.kron(S.contiguous(), K.contiguous())
        return block

    def build_Sinv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (S^{-1}, logdet S). Structured per left_hand; falls back to a
        dense regularized inverse when jitter > 0 (jitter breaks the structure).
        """
        log_S = self.log_S
        d = self.k * self.c

        if self.jitter > 0:
            S = self.build_S()
            abs_jitter = self.jitter * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
            rel_jitter = self.jitter * torch.diag_embed(torch.diagonal(S))
            S = S + abs_jitter + rel_jitter
            L = torch.linalg.cholesky(S)
            Sinv = torch.cholesky_inverse(L)
            logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
            return Sinv, logdet

        match self.left_hand:

            case "iid":
                # S = e^{log_S} I_d
                Sinv = torch.exp(-log_S) * torch.eye(d, dtype = self.dtype, device = self.device)
                logdet = d * log_S
                return Sinv, logdet

            case "diag":
                Sinv = torch.diag(torch.exp(-log_S))
                logdet = torch.sum(log_S)
                return Sinv, logdet

            case "full":
                S = torch.linalg.matrix_exp(log_S + log_S.T)
                L = torch.linalg.cholesky(S)
                Sinv = torch.cholesky_inverse(L)
                logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
                return Sinv, logdet

            case "fa":
                # Woodbury:
                # S = Q Λ Q' + Ψ,  Ψ diagonal, Q (d×q), q < d.
                #   S^{-1} = Ψ^{-1} - Ψ^{-1} Q (Λ^{-1} + Q'Ψ^{-1}Q)^{-1} Q'Ψ^{-1}
                #   logdet S = logdet Ψ + logdet Λ + logdet(Λ^{-1} + Q'Ψ^{-1}Q)
                q = self.n_axes
                M          = log_S[: d * q].reshape(d, q)
                log_Lambda = log_S[d * q : d * q + q]
                log_Psi    = log_S[d * q + q :]

                Q, _ = torch.linalg.qr(M)              # (d, q), same Q as build_S

                Psi_inv = torch.exp(-log_Psi)          # (d,), Ψ diagonal => free

                # capacitance C = Λ^{-1} + Q'Ψ^{-1}Q   (q×q, the only dense solve)
                QtPsiInv = Q.T * Psi_inv                # (q, d), broadcasts Ψ^{-1}
                C = torch.diag(torch.exp(-log_Lambda)) + QtPsiInv @ Q
                Lc = torch.linalg.cholesky(C)

                # S^{-1} = diag(Ψ^{-1}) - (Ψ^{-1}Q) C^{-1} (Q'Ψ^{-1})
                PsiInvQ = Q * Psi_inv.unsqueeze(1)      # (d, q) = Ψ^{-1} Q
                W = torch.cholesky_solve(PsiInvQ.T, Lc) # (q, d) = C^{-1} Q'Ψ^{-1}
                Sinv = torch.diag(Psi_inv) - PsiInvQ @ W

                # logdet S = Σlog Ψ + Σlog Λ + logdet C
                logdet_C = 2.0 * torch.sum(torch.log(torch.diagonal(Lc)))
                logdet = torch.sum(log_Psi) + torch.sum(log_Lambda) + logdet_C

                return Sinv, logdet

            case _:
                raise ValueError(f"unsupported left hand type: {self.left_hand}")

    def build_Kinv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (K^{-1}, logdet K). For str, both are precomputed once (cached)
        since K is constant across REML iterations.
        """
        match self.right_hand:

            case "iid":
                return (
                    torch.eye(self.L, dtype = self.dtype, device = self.device),
                    torch.zeros((), dtype = self.dtype, device = self.device),
                )
            
            case "ar":
                # K = exp(-ar D), depends on the trained ar: no choice, factor each step.
                ar = torch.exp(self.log_ar)
                K = torch.exp(-ar * self.distance)
                Lk = torch.linalg.cholesky(K)
                Kinv = torch.cholesky_inverse(Lk)
                logdet = 2.0 * torch.sum(torch.log(torch.diagonal(Lk)))
                return Kinv, logdet
            
            case "het":
                # K = diag(exp(V h)), h has a fixed 0 reference (first column)
                h = torch.cat([torch.zeros(1, dtype = self.dtype, device = self.device), self.log_h])
                diag = self.V @ h                      # log-variances
                Kinv = torch.diag(torch.exp(-diag))
                logdet = torch.sum(diag)
                return Kinv, logdet

            case "str":
                # K constant => invert + logdet once, then cache.
                if not hasattr(self, "Kinv"):
                    Lk = torch.linalg.cholesky(self.covariance)
                    self.Kinv = torch.cholesky_inverse(Lk)
                    self.logdet_K = 2.0 * torch.sum(torch.log(torch.diagonal(Lk)))
                return self.Kinv, self.logdet_K
            
            case _:
                raise ValueError(f"unsupported right hand type: {self.right_hand}")
        
    def varmeth_inv(self) -> Callable:
        """
        Return the closure producing this effect's inverse block and logdet:
            () -> (S^{-1} ⊗ K^{-1},  L·logdet S + d·logdet K)
        Reads live parameters at call time (in-place optimizer updates).
        """
        d = self.k * self.c

        def block_inv() -> tuple[torch.Tensor, torch.Tensor]:
            Sinv, logdet_S = self.build_Sinv()
            Kinv, logdet_K = self.build_Kinv()
            Ginv_e = torch.kron(Sinv.contiguous(), Kinv.contiguous())
            logdet_Ge = self.L * logdet_S + d * logdet_K
            return Ginv_e, logdet_Ge

        return block_inv

    def format_variance(self) -> None:
        """
        Format the estimated variance structure into a user-facing object.

        The formatted object is stored in `self.variance`.

        Conventions:
            - left_hand="iid"  -> sigma is a scalar
            - left_hand="diag" -> sigma is a diagonal covariance matrix
            - left_hand="full" -> sigma is a full covariance matrix
            - left_hand="fa"   -> sigma is a full covariance matrix

            - right_hand="het" -> sigma becomes a list:
                one scalar / matrix per het modality, equal to Sigma * h[i]
        """
        with torch.no_grad():
            S = self.build_S().detach().cpu().numpy()

            fa_meta = None

            # --- left-hand Sigma ------------------------------------------------
            match self.left_hand:
                case "iid":
                    sigma_base = float(torch.exp(self.log_S).detach().cpu().numpy())

                case "diag" | "full":
                    sigma_base = S

                case "fa":
                    sigma_base = S

                    d = self.k * self.c
                    q = self.n_axes
                    log_S = self.log_S
                    M = log_S[: d * q].reshape(d, q)
                    log_Lambda = log_S[d * q : d * q + q]
                    log_Psi = log_S[d * q + q :]

                    # sign convention: make diag(R) >= 0 (reporting only)
                    Q, R = torch.linalg.qr(M)
                    s = torch.sign(torch.diagonal(R))
                    s = torch.where(s == 0, torch.ones_like(s), s)
                    Q = Q * s

                    Lambda = torch.exp(log_Lambda)
                    Psi = torch.exp(log_Psi)

                    # canonical axis order: decreasing Lambda
                    order = torch.argsort(Lambda, descending=True)
                    Lambda = Lambda[order]
                    Q = Q[:, order]

                    fa_meta = {
                        "n_axes": int(q),
                        "Q": Q.detach().cpu().numpy(),
                        "Lambda": Lambda.detach().cpu().numpy(),
                        "Psi": Psi.detach().cpu().numpy(),
                    }

                case _:
                    raise ValueError(f"unsupported left hand type: {self.left_hand}")

            labels = [
                {"response": resp, "component": comp}
                for resp, comp in self.components
            ]

            metadata = {
                "labels": labels,
            }

            if fa_meta is not None:
                metadata["fa"] = fa_meta

            effect = self.unit if self.unit is not None else "residual"

            # --- right-hand factor ----------------------------------------------
            match self.right_hand:

                case "iid" | "str":
                    sigma = sigma_base

                case "ar":
                    sigma = sigma_base
                    metadata = {
                        **metadata,
                        "ar": float(torch.exp(self.log_ar).detach().cpu().numpy()),
                    }

                case "het":
                    # multiplicative variance factors per non-intercept patsy column
                    self.h = np.exp(self.log_h.detach().cpu().numpy())

                    sigma = sigma_base

                    het_labels = [
                        {"column": col, "h": float(h_i)}
                        for col, h_i in zip(self.het_index, self.h)
                    ]

                    metadata = {
                        **metadata,
                        "het_formula": self.het_formula,
                        "het": het_labels,
                    }

                case _:
                    raise ValueError(f"unsupported right hand type: {self.right_hand}")

            self.variance = {
                "effect": effect,
                "left_hand": self.left_hand,
                "right_hand": self.right_hand,
                "sigma": sigma,
                "metadata": metadata,
            }

class Random(GaussianComponent):

    def __init__(
        self,
        unit: str,
        formula: str = "1",
        left_hand: Literal["iid", "diag", "full", "fa"] = "iid",
        right_hand: Literal["iid", "ar", "str", "het"] = "iid",
        covariance: None | np.ndarray = None,
        distance: None | np.ndarray = None,
        matrix_index: None | list = None,
        het_formula: None | str = None,
        n_axes: None | int = None,
        init: None | float | np.ndarray = None,
        jitter: float = 0,
    ):
        super().__init__(
            unit = unit,
            formula = formula,
            left_hand = left_hand,
            right_hand = right_hand,
            covariance = covariance,
            distance = distance,
            matrix_index = matrix_index,
            het_formula = het_formula,
            n_axes = n_axes,
            init = init,
            jitter = jitter,
        )

    def design(
        self,
        data: pd.DataFrame,
        responses: list[str],
        dtype = torch.float,
        device = "cpu",
    ) -> np.ndarray:
        """
        Confront the random effect to the actual data: read the dimensions,
        store the constant right-hand inputs, instantiate the parameters, and
        return the (single response) incidence matrix Z. The response replication
        (block-diagonal over responses) is performed by MixedModel.from_dataframe.

        Dimensions read here:
        - k: number of responses
        - c: number of formula components (columns of `formula`)
        - L: number of levels of `unit`, or number of original rows for residuals
        - n: number of rows (observations) before masking
        - q: number of columns of Z, i.e. c * L
        """
        self.device = device
        self.dtype = dtype
        self.responses = list(responses)
        self.k = len(self.responses)

        self.index = np.sort(data[self.unit].unique())  # the levels

        self.make_Z(data)             # -> self.Z, self.colnames, self.c, self.L
        if self.right_hand == "het":
            self.make_V(data)
        self.n, self.q = self.Z.shape

        # constant right-hand inputs (the variable parts of K live in build_K).
        # Reorder/subset the user matrix from `matrix_index` to the data levels.
        
        if self.distance is not None or self.covariance is not None:
            if self.matrix_index is None:
                raise ValueError("matrix_index must be provided alongside covariance/distance")
            
            pos = {lvl: j for j, lvl in enumerate(self.matrix_index)}

            try:
                perm = [pos[lvl] for lvl in self.index]

            except KeyError as e:
                raise ValueError(f"level {e} present in data is missing from matrix_index")
            
            ix = np.ix_(perm, perm)

            if self.distance is not None:
                self.distance = torch.as_tensor(np.asarray(self.distance)[ix], dtype=dtype, device=device)
            if self.covariance is not None:
                self.covariance = torch.as_tensor(np.asarray(self.covariance)[ix], dtype=dtype, device=device)

        self.init_varparams()         # -> self.varparams, self.log_S, (self.log_ar)
        self.uhat = torch.zeros(self.k * self.c * self.L, 1, dtype=dtype, device=device)
        return self.Z

    def make_Z(
        self,
        data: pd.DataFrame
    ) -> None:
        """
        Build the incidence matrix Z for this random effect.

        Columns are ordered ELEMENT-outer / LEVEL-inner:

            [ e0·l0, e0·l1, ..., e0·l(L-1), e1·l0, ... ]

        so that, once MixedModel block-diagonalizes Z over responses (response-outer),
        the random vector u is ordered response -> element -> level. This matches
        kron(S, K) with S ordered (response-outer, element-inner) and K over the
        levels.

        For residual structures (`unit=None`), Z is the identity over the
        original rows of the DataFrame. MixedModel applies missing-data masks.
        """

        Z_df = patsy.dmatrix(self.formula, data=data, return_type="dataframe")
        self.colnames = list(Z_df.columns)   # patsy element names
        Z_base = Z_df.to_numpy()

        n, c = Z_base.shape
        self.c = c
        self.L = len(self.index)

        self.Z = np.zeros((n, c * self.L))
        unit_array = data[self.unit].to_numpy()
        for i, level in enumerate(self.index):
            mask = unit_array == level
            for j in range(c):
                self.Z[mask, j * self.L + i] = Z_base[mask, j]

    def format_pred(
        self,
        uhat: torch.Tensor,
        pev: None | torch.Tensor = None,
    ) -> None:
        """
        Receive this effect's uhat slice (and optionally its PEV sub-matrix),
        copy it into the immutable per-effect uhat, and build the labelled table.
        Row order is response-outer, component-mid, unit-inner.
        """
        self.uhat.copy_(uhat.detach().reshape(self.uhat.shape))
        self.PEV = pev
        self._fitted = True

        vals = self.uhat.cpu().numpy().ravel()
        levels = self.index.tolist()
        rows = []
        idx = 0
        for resp in self.responses:
            for comp in self.colnames:
                for lvl in levels:
                    rows.append((lvl, resp, comp, float(vals[idx])))
                    idx += 1
        self.table = pd.DataFrame(rows, columns=["unit", "response", "component", "prediction"])
    
    def predict(
            self,
            matrix_index: list,
            covariance: None | np.ndarray = None,
            distance: None | np.ndarray = None,
        ) -> pd.DataFrame:
        """
        Kriging prediction of the random effect at new levels.

        `matrix_index` orders a complete covariance/distance over all levels
        (training + new). The effect is predicted by conditioning the trained
        BLUP on the across-level structure:

            u_pred = u_train @ K_train^{-1} @ K_{train,pred}

        with u_train the trained BLUP reshaped to (components x train levels).
        The left-hand factor Sigma cancels in the conditioning, so only the
        right-hand factor K matters. Returns a long-format DataFrame with one row
        per (unit, response, component).
        """
        if self.right_hand not in ("str", "ar"):
            raise ValueError(
                "Prediction is only available for a structured ('str') or "
                "autoregressive ('ar') right-hand term."
            )
        if not getattr(self, "_fitted", False):
            raise ValueError("The model must be fitted before predicting.")

        train_levels = self.index.tolist()
        missing = [lvl for lvl in train_levels if lvl not in set(matrix_index)]
        if missing:
            raise ValueError(
                f"training levels missing from matrix_index: {missing}. "
                "A complete covariance / distance over all levels is required."
            )

        pos = {lvl: j for j, lvl in enumerate(matrix_index)}
        train_pos = [pos[lvl] for lvl in train_levels]
        pred_levels = [lvl for lvl in matrix_index if lvl not in set(train_levels)]
        pred_pos = [pos[lvl] for lvl in pred_levels]

        with torch.no_grad():

            # Full across-level factor K over matrix_index, via build_K's kernel.
            if self.right_hand == "str":
                if covariance is None:
                    raise ValueError("`covariance` is required to predict with right_hand='str'.")
                K_full = self.build_K(
                    covariance=torch.as_tensor(np.asarray(covariance), dtype=self.dtype, device = self.device)
                )
            else:  # "ar": build_K rebuilds exp(-rho * distance) with the trained rho
                if distance is None:
                    raise ValueError("`distance` is required to predict with right_hand='ar'.")
                K_full = self.build_K(
                    distance=torch.as_tensor(np.asarray(distance), dtype=self.dtype, device = self.device)
                )

            K_train = K_full[train_pos][:, train_pos]
            K_pred = K_full[pred_pos][:, train_pos]

            # u_train: (components, train levels)
            # Per component: a_pred = K_pred @ K_train^-1 @ a_train
            u_train = self.uhat.reshape(self.k * self.c, self.L) 

            u_pred = (K_pred @ torch.linalg.solve(K_train, u_train.T)).T.cpu().numpy()

            rows = [
                (lvl, resp, comp, u_pred[i, j])
                for j, lvl in enumerate(pred_levels)
                for i, (resp, comp) in enumerate(self.components)
            ]
            return pd.DataFrame(rows, columns=["unit", "response", "component", "prediction"])

class Residual(GaussianComponent):

    def __init__(
        self,
        left_hand: Literal["iid", "diag", "full", "fa"] = "iid",
        right_hand: Literal["iid", "ar", "str", "het"] = "iid",
        covariance: None | np.ndarray = None,
        distance: None | np.ndarray = None,
        het_formula: None | str = None,
        n_axes: None | int = None,
        init: None | float | np.ndarray = None,
        jitter: float = 0,
    ):
        super().__init__(
            unit = None,
            formula ="1",
            left_hand = left_hand,
            right_hand = right_hand,
            covariance = covariance,
            distance = distance,
            matrix_index = None,
            het_formula = het_formula,
            n_axes = n_axes,
            init = init,
            jitter = jitter,
        )

    def varmeth_inv(self) -> Callable:
        def block(W: torch.Tensor | None = None):
            diagonal = (self.left_hand in ("iid", "diag")) and (self.right_hand in ("iid", "het"))

            if W is None:
                # no masking: R = R_tot = S⊗K, invert and logdet by Kronecker structure
                Sinv, logdet_S = self.build_Sinv()
                Kinv, logdet_K = self.build_Kinv()
                Rinv = torch.kron(Sinv.contiguous(), Kinv.contiguous())
                logdet_R = self.L * logdet_S + (self.k * self.c) * logdet_K
                return Rinv, logdet_R

            if diagonal:
                # masked but fully diagonal: selection commutes, logdet from the diagonal
                Sinv, _ = self.build_Sinv()
                Kinv, _ = self.build_Kinv()
                Rinv = W @ torch.kron(Sinv.contiguous(), Kinv.contiguous()) @ W.T
                logdet_R = -torch.sum(torch.log(torch.diagonal(Rinv)))
                return Rinv, logdet_R

            # masked and dense: form R then factor
            R = W @ self.varmeth()() @ W.T
            L = torch.linalg.cholesky(R)
            Rinv = torch.cholesky_inverse(L)
            logdet_R = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
            return Rinv, logdet_R
        return block
    
    def design(
        self,
        data: pd.DataFrame,
        responses: list[str],
        dtype = torch.float,
        device = "cpu",
    ) -> np.ndarray:
        """
        Confront the residual to the actual data: read the dimensions,
        store the constant right-hand inputs, instantiate the parameters, and
        return the (single response) incidence matrix W. The response replication
        (block-diagonal over responses) is performed by MixedModel.from_dataframe.
        """
        self.device = device
        self.dtype = dtype
        self.responses = list(responses)
        self.k = len(self.responses)

        self.index = np.arange(len(data))

        self.make_W()             # -> self.Z, self.colnames, self.c, self.L
        if self.right_hand == "het":
            self.make_V(data)
        self.n, self.q = self.W.shape

        # constant right-hand inputs (the variable parts of K live in build_K).
        # Reorder/subset the user matrix from `matrix_index` to the data levels.
        if self.distance is not None or self.covariance is not None:

            if self.distance is not None:
                self.distance = torch.as_tensor(
                    np.asarray(self.distance),
                    dtype  = self.dtype,
                    device = self.device,
                )

            if self.covariance is not None:
                self.covariance = torch.as_tensor(
                    np.asarray(self.covariance),
                    dtype  = self.dtype,
                    device = self.device,
                )

        self.init_varparams()         # -> self.varparams, self.log_S, (self.log_ar)

        return self.W

    def make_W(self) -> None:
        """
        W is the identity over the original rows of the DataFrame,
        to which MixedModel applies missing-data masks.
        """
        self.colnames = ["Intercept"]
        self.c = 1
        self.L = len(self.index)
        self.W = np.eye(self.L)

    def format_residuals(
        self,
        residuals: torch.Tensor,
        Wtot: torch.Tensor,
    ) -> None:
        """
        Receive the model residuals from the fitted equations, store them, and build
        the labelled table.

        `Wtot` is the residual incidence matrix actually used by MixedModel after
        response-wise missing-data masking. Each row of Wtot corresponds to one
        retained observation in the stacked response vector y.

        Row order follows the native stacking order of y:
            response-outer, observed-row-inner

        The original observation index is reconstructed from the non-zero column of
        Wtot.
        """
        self.residuals = residuals

        W_np = Wtot.detach().cpu().numpy() if isinstance(Wtot, torch.Tensor) else np.asarray(Wtot)
        vals = residuals.detach().cpu().numpy().ravel()

        n_obs = self.W.shape[0]

        rows = []

        for i, value in enumerate(vals):
            nz = np.flatnonzero(W_np[i])

            if len(nz) != 1:
                raise ValueError(
                    "Each row of Wtot must contain exactly one non-zero entry."
                )

            global_col = int(nz[0])
            response_idx = global_col // n_obs
            observation_idx = global_col % n_obs

            rows.append(
                (
                    observation_idx,
                    self.responses[response_idx],
                    float(value),
                )
            )

        self.table = pd.DataFrame(
            rows,
            columns=["observation", "response", "residual"],
        )
