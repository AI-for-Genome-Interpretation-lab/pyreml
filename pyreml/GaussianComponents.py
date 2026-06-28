from typing import Literal, Callable
import numpy as np
from scipy.linalg import logm
import pandas as pd
import patsy
import torch
import torch.nn as nn
import warnings

LEFT = Literal["iid", "diag", "full", "fa", "bl_resp", "bl_form", "kr_resp", "kr_form"]
RIGHT = Literal["iid", "str", "het", "dist", "eucl", "ar_iso", "ar_ani"]

# right-hand families
_COORD_RIGHT = ("eucl", "ar_iso", "ar_ani")        # read coordinates from the frame
_DECAY_RIGHT = ("dist", "eucl", "ar_iso", "ar_ani")  # exp(-decay), rate(s) > 0


class GaussianComponent:

    def __init__(
        self,
        unit: str | None,
        formula: str = "1",
        left_hand: LEFT = "iid",
        right_hand: RIGHT = "iid",
        covariance: None | np.ndarray = None,
        distance: None | np.ndarray = None,
        coords: None | list[str] = None,
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
            - iid:  S = s² * I_d, a single scalar variance (default)
            - diag: a diagonal covariance matrix
            - full: a full covariance matrix
            - fa:   factor-analytic approximation of the full structure
            - bl_resp / bl_form: block-diagonal across responses / formula columns
                    (k blocks c×c, resp. c blocks k×k), each a full covariance
            - kr_resp / kr_form: the separable (proportional-block) counterparts,
                    S = diag(alpha) ⊗ Omega (resp. A ⊗ diag(omega)); the first
                    scaling factor is anchored to 1.

        Supported types for K (`right_hand`):
            - iid:  identity matrix, independent levels (default)
            - str:  known structure matrix, e.g. a kinship (see `covariance`)
            - het:  diagonal, log-linear in `het_formula`
            - dist: exp(-rho * D) over a supplied distance matrix `distance`
            - eucl: exp(-rho * ||dx||_2), distance built internally from `coords`
            - ar_iso / ar_ani: separable autoregressive decay from `coords`,
                    one shared rate (iso) or one rate per axis (ani); the levels
                    are taken on the complete integer grid spanned by `coords`.

        `covariance` is the known structure matrix for right_hand="str".
        `distance`   is the known distance matrix for right_hand="dist".
        `coords` lists the DataFrame columns holding the coordinates for the
            coordinate-based kernels (eucl / ar_iso / ar_ani). Each level of
            `unit` must carry a single coordinate (constant within the level) and
            no two levels may share the same coordinate (otherwise K is singular).
            For ar_iso / ar_ani the coordinates must be integer-valued.
        `matrix_index` lists the levels of `unit` in the order in which they
            appear in the rows/cols of `covariance` / `distance`.
        `het_formula` is the formula for right_hand="het".
        `n_axes` is the number of factorial axes for left_hand="fa".
        `init` is the initial value of S. For the matrix structures it is a
            (k*c) x (k*c) array; entries outside the retained block / Kronecker
            structure are ignored. For left_hand="iid" it is a scalar.
        """

        self.unit = unit
        self.formula = formula
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.covariance = covariance
        self.distance = distance
        self.coords = coords
        self.matrix_index = matrix_index
        self.het_formula = het_formula
        self.n_axes = n_axes
        self.init = init
        self.jitter = jitter

        # check "dist"
        if self.right_hand == "dist":
            if self.distance is None:
                raise ValueError("`distance` is required when right_hand = 'dist'.")
        elif self.distance is not None:
            raise ValueError("`distance` must be None when right_hand != 'dist'.")

        # check str
        if self.right_hand == "str":
            if self.covariance is None:
                raise ValueError("`covariance` is required when right_hand = 'str'.")
        elif self.covariance is not None:
            raise ValueError("`covariance` must be None when right_hand != 'str'.")

        # check coords (eucl / ar_iso / ar_ani)
        if self.right_hand in _COORD_RIGHT:
            if self.coords is None:
                raise ValueError(
                    "`coords` is required when right_hand in {'eucl', 'ar_iso', 'ar_ani'}."
                )
        elif self.coords is not None:
            raise ValueError(
                "`coords` must be None unless right_hand in {'eucl', 'ar_iso', 'ar_ani'}."
            )

        # check het
        if self.right_hand == "het":
            if self.het_formula is None:
                raise ValueError("`het_formula` is required when `right_hand='het'`.")
        elif self.het_formula is not None:
            raise ValueError("`het_formula` must be None when right_hand != 'het'.")
        
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
            raise ValueError("`n_axes` must be None when left_hand != 'fa'.")

    def init_varparams(self) -> None:
        """
        Instantiate the trainable variance parameters for this effect.

        Left-hand factor S, stored in its log-parameterization (d = k * c):
            - iid:  S = exp(log_S) * I_d              (log_S a scalar)
            - full: S = matrix_exp(log_S + log_S.T)   (log_S a (d, d) matrix)
            - diag: S = diag(exp(log_S))              (log_S a (d,) vector)
            - fa:   S = Q diag(Lambda) Q.T + diag(Psi)
            - bl_resp: log_S a (k, c, c) stack, each block matrix_exp(B + B.T)
            - bl_form: log_S a (c, k, k) stack, blocks laid formula-outer then permuted
            - kr_resp: log_S = [Omega_flat (c*c), log_alpha (k-1)]
                       S = diag([1, exp(log_alpha)]) ⊗ matrix_exp(B + B.T)
            - kr_form: log_S = [A_flat (k*k), log_omega (c-1)]
                       S = matrix_exp(B + B.T) ⊗ diag([1, exp(log_omega)])

        Right-hand factor K:
            - iid / str: no trainable parameter
            - dist / eucl / ar_iso: a single log_rho (rho = exp(log_rho) > 0)
            - ar_ani: a log_rho vector, one rate per coordinate axis
            - het: p-1 trainable log-ratios log_h. Requires make_V(data) beforehand.
        """
        self.varparams = []

        # components of the effect, response-outer / element-inner: matches S order
        self.components = [
            (resp, name) for resp in self.responses for name in self.colnames
        ]
        k, c, d = self.k, self.c, self.d

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
                    log_S0 = torch.as_tensor(np.real(logm(init)) / 2.0, dtype=self.dtype, device=self.device)
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

            case "bl_resp":
                # k blocks of c x c, each parameterized like `full`
                if self.init is None:
                    log_S0 = torch.zeros(k, c, c, dtype=self.dtype, device=self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    blocks = [
                        np.real(logm(init[r * c:(r + 1) * c, r * c:(r + 1) * c])) / 2.0
                        for r in range(k)
                    ]
                    log_S0 = torch.as_tensor(np.stack(blocks), dtype=self.dtype, device=self.device)
                self.log_S = nn.Parameter(log_S0)
                n_left = k * c * (c + 1) // 2

            case "bl_form":
                # c blocks of k x k (one per formula column)
                if self.init is None:
                    log_S0 = torch.zeros(c, k, k, dtype=self.dtype, device=self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    blocks = []
                    for e in range(c):
                        idx = [r * c + e for r in range(k)]
                        blocks.append(np.real(logm(init[np.ix_(idx, idx)])) / 2.0)
                    log_S0 = torch.as_tensor(np.stack(blocks), dtype=self.dtype, device=self.device)
                self.log_S = nn.Parameter(log_S0)
                n_left = c * k * (k + 1) // 2

            case "kr_resp":
                # S = diag([1, exp(log_alpha)]) ⊗ Omega, Omega (c x c) full
                if self.init is None:
                    Bflat = torch.zeros(c * c, dtype=self.dtype, device=self.device)
                    log_alpha = torch.zeros(k - 1, dtype=self.dtype, device=self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    Omega0 = init[:c, :c]
                    Bflat = torch.as_tensor(
                        (np.real(logm(Omega0)) / 2.0).reshape(-1), dtype=self.dtype, device=self.device
                    )
                    ratios = [
                        np.trace(init[r * c:(r + 1) * c, r * c:(r + 1) * c]) / np.trace(Omega0)
                        for r in range(1, k)
                    ]
                    log_alpha = torch.as_tensor(
                        np.log(ratios) if ratios else np.zeros(0), dtype=self.dtype, device=self.device
                    )
                self.log_S = nn.Parameter(torch.cat([Bflat, log_alpha]))
                n_left = c * (c + 1) // 2 + (k - 1)

            case "kr_form":
                # S = A ⊗ diag([1, exp(log_omega)]), A (k x k) full
                if self.init is None:
                    Bflat = torch.zeros(k * k, dtype=self.dtype, device=self.device)
                    log_omega = torch.zeros(c - 1, dtype=self.dtype, device=self.device)
                else:
                    init = np.asarray(self.init, dtype=float)
                    idx0 = [r * c + 0 for r in range(k)]
                    A0 = init[np.ix_(idx0, idx0)]
                    Bflat = torch.as_tensor(
                        (np.real(logm(A0)) / 2.0).reshape(-1), dtype=self.dtype, device=self.device
                    )
                    ratios = []
                    for e in range(1, c):
                        idx_e = [r * c + e for r in range(k)]
                        ratios.append(np.trace(init[np.ix_(idx_e, idx_e)]) / np.trace(A0))
                    log_omega = torch.as_tensor(
                        np.log(ratios) if ratios else np.zeros(0), dtype=self.dtype, device=self.device
                    )
                self.log_S = nn.Parameter(torch.cat([Bflat, log_omega]))
                n_left = k * (k + 1) // 2 + (c - 1)

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

            case "dist":
                if self.distance is None:
                    raise ValueError("a distance matrix `distance` must be provided for right_hand='dist'")
                self.log_rho = nn.Parameter(torch.zeros((), dtype=self.dtype, device=self.device))
                self.varparams.append({
                    "effect": self.unit if self.unit is not None else "residual",
                    "element": "rho",
                    "tensor": self.log_rho,
                    "structure": "dist",
                })
                n_right = 1

            case "eucl" | "ar_iso":
                if not hasattr(self, "coord_dim"):
                    raise ValueError(
                        "make_coords(data) must be called before init_varparams for "
                        f"right_hand='{self.right_hand}'."
                    )
                self.log_rho = nn.Parameter(torch.zeros((), dtype=self.dtype, device=self.device))
                self.varparams.append({
                    "effect": self.unit if self.unit is not None else "residual",
                    "element": "rho",
                    "tensor": self.log_rho,
                    "structure": self.right_hand,
                })
                n_right = 1

            case "ar_ani":
                if not hasattr(self, "coord_dim"):
                    raise ValueError(
                        "make_coords(data) must be called before init_varparams for right_hand='ar_ani'."
                    )
                self.log_rho = nn.Parameter(torch.zeros(self.coord_dim, dtype=self.dtype, device=self.device))
                self.varparams.append({
                    "effect": self.unit if self.unit is not None else "residual",
                    "element": "rho",
                    "tensor": self.log_rho,
                    "index": list(self.coords),
                    "structure": "ar_ani",
                })
                n_right = self.coord_dim

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

        self.V = torch.tensor(V_df.to_numpy(), dtype=self.dtype, device=self.device)
        self.het_index = rest
        self.n_het = len(rest)

    def make_coords(self, data: pd.DataFrame, checkerboard: bool = False) -> None:
        """
        Read the coordinates for the coordinate-based kernels.

        The level contract is the SAME for both regimes and is checked here:
            - intra-level constancy: every row of a level carries one coordinate,
            - inter-level uniqueness: no two levels share a coordinate (else the
              across-level kernel is singular).
        A level is a level of `unit` (grouped) or an observation (residual).

        checkerboard=False (eucl) — the levels stay the observed positions; the
          across-level factor is dense over them. Sets self.coords_levels
          ((L, axes) tensor) and self.coord_dim.

        checkerboard=True (ar_iso / ar_ani) — the validated integer positions are
          embedded in the COMPLETE integer grid they span (one cell per integer
          position on every axis), so that K factors exactly as a Kronecker
          product of per-axis AR1 kernels. The level incidence (Z for a grouped
          effect, W for the residual) is re-laid over the grid cells; cells with
          no level become legitimate empty columns. Sets self.axis_grids,
          self.index (grid cells, axis-0-outer / axis-(d-1)-inner), self.L,
          self.coord_dim and self.level_cell (grid cell of each validated level).
        """
        if self.coords is None:
            raise ValueError("`coords` is required for the coordinate-based kernels.")

        # one coordinate per level, with the shared contract checks
        if self.unit is None:
            P = data[self.coords].to_numpy(dtype=float)              # per observation
        else:
            g = data.groupby(self.unit, sort=False)[self.coords]
            if (g.nunique() > 1).to_numpy().any():
                raise ValueError("`coords` must be constant within each level of `unit`.")
            P = g.first().reindex(self.index).to_numpy(dtype=float)  # per unit level

        if np.unique(P, axis=0).shape[0] != P.shape[0]:
            raise ValueError(
                "`coords` must be unique across levels: duplicated coordinates make "
                "the across-level kernel singular."
            )

        self.coord_dim = P.shape[1]

        if not checkerboard:
            # eucl: the levels remain the observed positions
            self.coords_levels = torch.as_tensor(P, dtype=self.dtype, device=self.device)
            return

        # --- ar: complete the integer grid spanned by the validated levels ------
        Pint = np.rint(P).astype(int)                               # (L_lvl, axes)
        axes = Pint.shape[1]
        starts, stops = Pint.min(axis=0), Pint.max(axis=0)
        self.axis_grids = [
            torch.arange(int(starts[a]), int(stops[a]) + 1, dtype=self.dtype, device=self.device)
            for a in range(axes)
        ]
        sizes = [int(stops[a] - starts[a] + 1) for a in range(axes)]
        L_grid = int(np.prod(sizes))

        # grid cells, axis-0-outer / axis-(d-1)-inner (matches the Kronecker order)
        mesh = np.meshgrid(*[grd.cpu().numpy() for grd in self.axis_grids], indexing="ij")
        grid_cells = np.stack([m.ravel() for m in mesh], axis=1)    # (L_grid, axes)

        # grid cell of each validated level (row-major over the axis grids)
        strides = np.array([int(np.prod(sizes[a + 1:])) for a in range(axes)])
        self.level_cell = ((Pint - starts) * strides).sum(axis=1)  # (L_lvl,)

        # embed the level incidence (Z grouped, W residual) into the grid columns;
        # uniqueness guarantees at most one level per cell, the rest stay empty.
        src = self.Z if self.unit is not None else self.W          # (n, c * L_lvl)
        L_lvl = len(self.level_cell)
        grid_incidence = np.zeros((src.shape[0], self.c * L_grid))
        for j in range(self.c):
            grid_incidence[:, j * L_grid + self.level_cell] = src[:, j * L_lvl:(j + 1) * L_lvl]

        if self.unit is not None:
            self.Z = grid_incidence
        else:
            self.W = grid_incidence

        self.index = grid_cells
        self.L = L_grid

    def build_S(self) -> torch.Tensor:
        """
        Rebuild the left-hand factor S from its log-parameterization.
        Differentiable in self.log_S.
        """
        log_S = self.log_S
        k, c, d = self.k, self.c, self.d

        match self.left_hand:

            case "iid":
                return torch.exp(log_S) * torch.eye(d, dtype=self.dtype, device=self.device)
            
            case "full":
                return torch.linalg.matrix_exp(log_S + log_S.T)
            
            case "diag":
                return torch.diag(torch.exp(log_S))
            
            case "fa":
                q = self.n_axes
                M = log_S[: d * q].reshape(d, q)
                log_Lambda = log_S[d * q: d * q + q]
                log_Psi = log_S[d * q + q:]

                Q, _ = torch.linalg.qr(M)
                Lambda = torch.exp(log_Lambda)
                Psi = torch.exp(log_Psi)
                return (Q * Lambda) @ Q.T + torch.diag(Psi)

            case "bl_resp":
                blocks = torch.linalg.matrix_exp(log_S + log_S.transpose(-1, -2))  # (k, c, c)
                return torch.block_diag(*blocks)

            case "bl_form":
                blocks = torch.linalg.matrix_exp(log_S + log_S.transpose(-1, -2))  # (c, k, k)
                S_form = torch.block_diag(*blocks)                                  # formula-outer
                perm = torch.arange(self.d, device=self.device).reshape(self.c, self.k).T.reshape(-1)
                return S_form[perm][:, perm]

            case "kr_resp":
                Bflat = log_S[: c * c].reshape(c, c)
                log_alpha = log_S[c * c:]
                Omega = torch.linalg.matrix_exp(Bflat + Bflat.T)
                alpha = torch.cat([torch.ones(1, dtype=self.dtype, device=self.device), torch.exp(log_alpha)])
                return torch.kron(torch.diag(alpha), Omega)

            case "kr_form":
                Bflat = log_S[: k * k].reshape(k, k)
                log_omega = log_S[k * k:]
                Alpha = torch.linalg.matrix_exp(Bflat + Bflat.T)
                omega = torch.cat([torch.ones(1, dtype=self.dtype, device=self.device), torch.exp(log_omega)])
                return torch.kron(Alpha, torch.diag(omega))

            case _:
                raise NotImplementedError(f"left_hand='{self.left_hand}' is not implemented.")

    def build_K(
        self,
        covariance: None | torch.Tensor = None,
        distance: None | torch.Tensor = None,
        coords: None | torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Build the right-hand factor K. Differentiable when the structure carries
        trainable parameters (dist, eucl, ar_iso, ar_ani, het), constant
        otherwise (iid, str).

        `covariance` / `distance` / `coords` default to the training inputs stored
        on the effect; passing new ones (over a superset of levels) lets `predict`
        reuse the same kernel logic for kriging.
        """
        covariance = self.covariance if covariance is None else covariance
        distance = self.distance if distance is None else distance

        match self.right_hand:

            case "iid":
                return torch.eye(self.L, dtype=self.dtype, device=self.device)
            
            case "dist":
                rho = torch.exp(self.log_rho)
                return torch.exp(-rho * distance)
            
            case "str":
                return covariance
            
            case "het":
                h = torch.cat([torch.zeros(1, dtype=self.dtype, device=self.device), self.log_h])
                return torch.diag(torch.exp(self.V @ h))
            
            case "eucl":
                P = self.coords_levels if coords is None else torch.as_tensor(coords, dtype=self.dtype, device=self.device)
                diff = P[:, None, :] - P[None, :, :]
                D = torch.sqrt((diff ** 2).sum(-1))
                return torch.exp(-torch.exp(self.log_rho) * D)

            case "ar_iso" | "ar_ani":
                rho = torch.exp(self.log_rho)                # scalar (iso) or (axes,) (ani)

                if coords is not None:
                    # prediction path: dense separable kernel over the supplied coords
                    P = torch.as_tensor(coords, dtype=self.dtype, device=self.device)
                    absdiff = (P[:, None, :] - P[None, :, :]).abs()    # (M, M, axes)
                    return torch.exp(-(absdiff * rho).sum(-1))         # rho broadcasts (iso/ani)

                # training path: K = ⊗_a exp(-rho_a |dx_a|) over the per-axis grids
                K = None
                for a, g in enumerate(self.axis_grids):
                    rho_a = rho[a] if self.right_hand == "ar_ani" else rho
                    K_a = torch.exp(-rho_a * (g[:, None] - g[None, :]).abs())
                    K = K_a if K is None else torch.kron(K, K_a)
                return K

            case _:
                raise ValueError(f"unsupported right hand type: {self.right_hand}")
        
    def varmeth(self) -> Callable:
        """
        Return the unitary function for this effect: a closure that builds this
        effect's block S ⊗ K.
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
        Return (S^{-1}, logdet S). Structured per left_hand; falls back to a dense
        regularized inverse when jitter > 0 (jitter breaks the structure).
        """
        log_S = self.log_S
        k, c, d = self.k, self.c, self.d

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

            case "bl_resp":
                # block-by-block inverse and logdet (k blocks of c x c)
                blocks = torch.linalg.matrix_exp(log_S + log_S.transpose(-1, -2))
                invs, logdet = [], log_S.new_zeros(())
                for b in blocks:
                    Lb = torch.linalg.cholesky(b)
                    invs.append(torch.cholesky_inverse(Lb))
                    logdet = logdet + 2.0 * torch.sum(torch.log(torch.diagonal(Lb)))
                return torch.block_diag(*invs), logdet

            case "bl_form":
                blocks = torch.linalg.matrix_exp(log_S + log_S.transpose(-1, -2))
                invs, logdet = [], log_S.new_zeros(())
                for b in blocks:
                    Lb = torch.linalg.cholesky(b)
                    invs.append(torch.cholesky_inverse(Lb))
                    logdet = logdet + 2.0 * torch.sum(torch.log(torch.diagonal(Lb)))
                Sinv_form = torch.block_diag(*invs)
                perm = torch.arange(self.d, device=self.device).reshape(self.c, self.k).T.reshape(-1)
                return Sinv_form[perm][:, perm], logdet

            case "kr_resp":
                # S = diag(alpha) ⊗ Omega  =>  S^-1 = diag(1/alpha) ⊗ Omega^-1
                Bflat = log_S[: c * c].reshape(c, c)
                log_alpha = log_S[c * c:]
                Omega = torch.linalg.matrix_exp(Bflat + Bflat.T)
                alpha = torch.cat([torch.ones(1, dtype=self.dtype, device=self.device), torch.exp(log_alpha)])

                Lom = torch.linalg.cholesky(Omega)
                Omega_inv = torch.cholesky_inverse(Lom)
                logdet_Om = 2.0 * torch.sum(torch.log(torch.diagonal(Lom)))

                Sinv = torch.kron(torch.diag(1.0 / alpha), Omega_inv)
                logdet = c * torch.sum(torch.log(alpha)) + k * logdet_Om
                return Sinv, logdet

            case "kr_form":
                # S = A ⊗ diag(omega)  =>  S^-1 = A^-1 ⊗ diag(1/omega)
                Bflat = log_S[: k * k].reshape(k, k)
                log_omega = log_S[k * k:]
                A = torch.linalg.matrix_exp(Bflat + Bflat.T)
                omega = torch.cat([torch.ones(1, dtype=self.dtype, device=self.device), torch.exp(log_omega)])

                La = torch.linalg.cholesky(A)
                A_inv = torch.cholesky_inverse(La)
                logdet_A = 2.0 * torch.sum(torch.log(torch.diagonal(La)))

                Sinv = torch.kron(A_inv, torch.diag(1.0 / omega))
                logdet = c * logdet_A + k * torch.sum(torch.log(omega))
                return Sinv, logdet

            case _:
                raise ValueError(f"unsupported left hand type: {self.left_hand}")

    def build_Kinv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (K^{-1}, logdet K). For str, both are precomputed once (cached);
        the dense decay kernels are factored every step; the autoregressive
        kernels use the closed-form per-axis AR(1) inverse combined by Kronecker.
        """
        match self.right_hand:

            case "iid":
                return (
                    torch.eye(self.L, dtype=self.dtype, device=self.device),
                    torch.zeros((), dtype=self.dtype, device=self.device),
                )
            
            case "dist" | "eucl":
                # K depends on the trained rate: factor each step (dense).
                K = self.build_K()
                Lk = torch.linalg.cholesky(K)
                Kinv = torch.cholesky_inverse(Lk)
                logdet = 2.0 * torch.sum(torch.log(torch.diagonal(Lk)))
                return Kinv, logdet

            case "ar_iso" | "ar_ani":
                # Separable grid: K = ⊗_a K_a. Each per-axis AR(1) factor has a
                # closed-form tridiagonal inverse and log-determinant, so the
                # across-level inverse needs no factorization:
                #   K^-1 = ⊗_a K_a^-1,  logdet K = Σ_a (L / L_a) logdet K_a.
                rho = torch.exp(self.log_rho)
                sizes = [len(g) for g in self.axis_grids]
                Kinv = None
                logdet = self.log_rho.new_zeros(())
                for a in range(len(self.axis_grids)):
                    rho_a = rho[a] if self.right_hand == "ar_ani" else rho

                    # Closed-form inverse and log-determinant of a regular AR(1) factor
                    # K_ij = phi^{|i-j|}, phi = exp(-rho). The inverse is tridiagonal and the
                    # log-determinant is closed-form, so no factorization is needed:
                    #     K^{-1} = 1/(1-phi^2) * tridiag(-phi, [1, 1+phi^2, ..., 1+phi^2, 1], -phi)
                    #     logdet K = (L-1) log(1 - phi^2)

                    L = sizes[a]
                    phi_a = torch.exp(-rho_a)
                    if L == 1:
                        return (torch.ones(1, 1, dtype=self.dtype, device=self.device),
                                self.log_rho.new_zeros(()))
                    denom = 1.0 - phi_a * phi_a
                    edge = (1.0 / denom).reshape(1)
                    if L == 2:
                        main = torch.cat([edge, edge])
                    else:
                        inner = ((1.0 + phi_a * phi_a) / denom).reshape(1).expand(L - 2)
                        main = torch.cat([edge, inner, edge])
                    off = (-phi_a / denom).reshape(1).expand(L - 1)
                    Kinv_a = torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)
                    logdet_a = (L - 1) * torch.log(denom)

                    logdet = logdet + (self.L // sizes[a]) * logdet_a
                    Kinv = Kinv_a if Kinv is None else torch.kron(Kinv, Kinv_a)
                    
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
        """
        d = self.d

        def block_inv() -> tuple[torch.Tensor, torch.Tensor]:
            Sinv, logdet_S = self.build_Sinv()
            Kinv, logdet_K = self.build_Kinv()
            Ginv_e = torch.kron(Sinv.contiguous(), Kinv.contiguous())
            logdet_Ge = self.L * logdet_S + d * logdet_K
            return Ginv_e, logdet_Ge

        return block_inv

    def format_variance(self) -> None:
        """
        Format the estimated variance structure into a user-facing object stored
        in `self.variance`. `metadata["rho"]` is a list in every case (empty for
        structures with no estimated rate).
        """
        with torch.no_grad():
            S = self.build_S().detach().cpu().numpy()

            fa_meta = None

            # --- left-hand Sigma ------------------------------------------------
            match self.left_hand:
                case "iid":
                    sigma_base = float(torch.exp(self.log_S).detach().cpu().numpy())

                case "diag" | "full" | "bl_resp" | "bl_form" | "kr_resp" | "kr_form":
                    sigma_base = S

                case "fa":
                    sigma_base = S

                    d = self.d
                    q = self.n_axes
                    log_S = self.log_S
                    M = log_S[: d * q].reshape(d, q)
                    log_Lambda = log_S[d * q: d * q + q]
                    log_Psi = log_S[d * q + q:]

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

                case "dist" | "eucl" | "ar_iso":
                    sigma = sigma_base
                    metadata = {
                        **metadata,
                        "rho": float(torch.exp(self.log_rho).detach().cpu().numpy())
                    }

                case "ar_ani":
                    sigma = sigma_base
                    rho = torch.exp(self.log_rho).detach().cpu().numpy()
                    metadata = {
                        **metadata,
                        "rho": [float(x) for x in rho],
                        "coords": list(self.coords),
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
        left_hand: LEFT = "iid",
        right_hand: RIGHT = "iid",
        covariance: None | np.ndarray = None,
        distance: None | np.ndarray = None,
        coords: None | list[str] = None,
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
            coords = coords,
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
        if self.right_hand == "eucl":
            self.make_coords(data, checkerboard=False)
        elif self.right_hand in ("ar_iso", "ar_ani"):
            # ar lives on a regular integer grid: check, convert, then fill the grid
            P = data[self.coords].to_numpy(dtype=float)
            if not np.allclose(P, np.rint(P)):
                raise ValueError(
                    "right_hand in {'ar_iso', 'ar_ani'} requires integer-valued `coords` "
                    "(a regular grid). Use 'eucl' for arbitrary real coordinates."
                )
            self.make_coords(data, checkerboard=True)   # rebuilds self.Z over the grid
        self.n, self.q = self.Z.shape
        self.d = self.k * self.c

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

        self.init_varparams()         # -> self.varparams, self.log_S, (self.log_rho)
        self.uhat = torch.zeros(self.d * self.L, 1, dtype=dtype, device=device)
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
        coords: None | np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Kriging prediction of the random effect at new levels.

        `matrix_index` orders the across-level inputs over all levels (training +
        new). The effect is predicted by conditioning the trained BLUP:

            u_pred = u_train @ K_train^{-1} @ K_{train,pred}

        The left-hand factor S cancels in the conditioning, so only the right-hand
        factor K matters. The across-level input depends on the structure:
            - str:  `covariance` over matrix_index,
            - dist: `distance` over matrix_index,
            - eucl/ar_iso/ar_ani: `coords` array (len(matrix_index) x axes),
              aligned to matrix_index.
        """
        if self.right_hand not in ("str", *_DECAY_RIGHT):
            raise ValueError(
                "Prediction is only available for 'str' or a decay kernel "
                "('dist', 'eucl', 'ar_iso', 'ar_ani')."
            )
        if not getattr(self, "_fitted", False):
            raise ValueError("The model must be fitted before predicting.")

        train_levels = self.index.tolist()
        missing = [lvl for lvl in train_levels if lvl not in set(matrix_index)]
        if missing:
            raise ValueError(
                f"training levels missing from matrix_index: {missing}. "
                "A complete input over all levels is required."
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
                    covariance=torch.as_tensor(np.asarray(covariance), dtype = self.dtype, device = self.device)
                )
            elif self.right_hand == "dist":
                if distance is None:
                    raise ValueError("`distance` is required to predict with right_hand='dist'.")
                K_full = self.build_K(
                    distance=torch.as_tensor(np.asarray(distance), dtype = self.dtype, device = self.device)
                )
            else:  # eucl / ar_iso / ar_ani
                if coords is None:
                    raise ValueError(
                        "`coords` is required to predict with right_hand in "
                        "{'eucl', 'ar_iso', 'ar_ani'}."
                    )
                K_full = self.build_K(
                    coords=torch.as_tensor(np.asarray(coords), dtype=self.dtype, device=self.device)
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
        left_hand: LEFT = "iid",
        right_hand: RIGHT = "iid",
        covariance: None | np.ndarray = None,
        distance: None | np.ndarray = None,
        coords: None | list[str] = None,
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
            coords=coords,
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
                logdet_R = self.L * logdet_S + (self.d) * logdet_K
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

        self.make_W()             # -> self.W, self.colnames, self.c, self.L
        if self.right_hand == "het":
            self.make_V(data)
        if self.right_hand == "eucl":
            self.make_coords(data, checkerboard=False)
        elif self.right_hand in ("ar_iso", "ar_ani"):
            P = data[self.coords].to_numpy(dtype=float)
            if not np.allclose(P, np.rint(P)):
                raise ValueError(
                    "right_hand in {'ar_iso', 'ar_ani'} requires integer-valued `coords` "
                    "(a regular grid). Use 'eucl' for arbitrary real coordinates."
                )
            self.make_coords(data, checkerboard=True)   # rebuilds self.W over the grid
        self.n, self.q = self.W.shape
        self.d = self.k * self.c

        if self.distance is not None or self.covariance is not None:
            if self.distance is not None:
                self.distance = torch.as_tensor(np.asarray(self.distance), dtype=self.dtype, device=self.device)
            if self.covariance is not None:
                self.covariance = torch.as_tensor(np.asarray(self.covariance), dtype=self.dtype, device=self.device)

        self.init_varparams()         # -> self.varparams, self.log_S, (self.log_rho)

        return self.W

    def make_W(self) -> None:
        """
        W is the identity over the original rows of the DataFrame.
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
