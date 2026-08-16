"""
Structured decomposition of the mixed-model covariance, and its two solve paths.

Three objects, on two lifetimes.

`Variance` is persistent: built once from the design, it describes the structure
and owns nothing of the current iterate. `DirectSolve` and `Capacitance` are
ephemeral: one is produced per loss evaluation, carries that evaluation's
factorization and projections, and dies with the optimization step.

Both optimization axes read from the same decomposition, which is why they are
neither nested nor redundant: the structured forward assembles V (or applies
Rinv) from it, the analytic backward reads gradient grains from it. Forward and
backward are two independent consumers of one shared structure.

The whole thing rests on one property of the incidence built by
MixedModel.from_dataframe: Z_e is one-hot in *levels* — each row loads a single
level l(i), while carrying c arbitrary continuous component values. This is
structural, imposed by Random.make_Z, and holds for random regression, for
coordinate kernels and for the matrix_index expansion alike. It lets each effect
contribute

    V_e = (F_e S_e F_e') ⊙ K_e^obs,    K_e^obs[i,j] = K_e[l(i), l(j)],

with F_e (n, d_e) the response-block-diagonal component values. Neither the
(d_e·L_e)² Kronecker block nor the ZGZ' product is ever formed.

The gradient of the REML loss wrt a variance parameter theta is tr(A dV/dtheta)
with

    A = V⁻¹ − V⁻¹rr'V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹.

Both paths compute that same three-term object, in different spaces: the direct
path forms A itself (n×n), the SMW path forms its projection Z'AZ (q×q) and
never touches V. `Solve._three_terms` carries the shared algebra.
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional
import torch

@dataclass
class Block:
    """
    One effect's contribution to the covariance, in factored form.

    - F: (n, d) response-block-diagonal component values
    - lev: (n,) level index of each observation, in the stacked ordering
    - k_obs: K[lev][:, lev], frozen once for constant-K hands (iid, str), None
      when K is trainable and must be gathered on every call
    - comp: the Random or Residual component owning the variance parameters
    - is_residual: the residual is a block like any other for V, but the SMW
      path treats it apart; the flag makes that explicit rather than relying on
      its position in the list
    """
    F: torch.Tensor
    lev: torch.Tensor
    k_obs: Optional[torch.Tensor]
    comp: object
    is_residual: bool = False

    def K_obs(self) -> torch.Tensor:
        """K restricted to the observed stacking, frozen or gathered."""
        if self.k_obs is not None:
            return self.k_obs
        K = self.comp.build_K()
        return K[self.lev][:, self.lev]

    def term(self) -> torch.Tensor:
        """This block's contribution to V: (F S F') ⊙ K_obs."""
        S = self.comp.build_S_full()
        return (self.F @ S @ self.F.T) * self.K_obs()

    def grain(self, A: torch.Tensor) -> list:
        """
        Gradient constants wrt this block's variance parameters, at the current
        point, from the model-level gradient matrix A:

            grain_S = F'(A ⊙ K_obs)F                 (dl/dS)
            grain_K = (A ⊙ F S F')[lev][:, lev]      (dl/dK, trainable K only)

        Returns the (tensor, grain) pairs whose inner products form the ghost
        loss: the tensor carries the parameter graph, the grain is a constant.
        Pairing against build_S_full reproduces dl/dtheta exactly, jitter
        included, since the relative-jitter term lives in that same graph.
        """
        S = self.comp.build_S_full()

        if self.k_obs is not None:
            K, K_obs = None, self.k_obs
        else:
            K = self.comp.build_K()
            K_obs = K[self.lev][:, self.lev]

        pairs = [(S, (self.F.T @ (A * K_obs.detach()) @ self.F).detach())]

        if K is not None:
            grain_K = (A * (self.F @ S.detach() @ self.F.T))[self.lev][:, self.lev]
            pairs.append((K, grain_K.detach()))

        return pairs

    def to(self, dtype: torch.dtype) -> None:
        """Cast the frozen tensors in place, following the model's working dtype."""
        self.F = self.F.to(dtype)
        if self.k_obs is not None:
            self.k_obs = self.k_obs.to(dtype)


@dataclass
class Embedding:
    """
    Residual structure of the SMW path.

    Zg and Xg lift the incidence and the design into the full (response × level)
    space, so that applying Rinv becomes a Kronecker multiply (Sinv ⊗ Kinv) and
    Rinv (n×n) is never formed. Exists only when the Kronecker identities hold:
    a Rtrick residual (fully diagonal) or balanced data (no missing cell). Its
    absence is what turns SMW off for good, in from_dataframe.
    """
    grid: torch.Tensor
    Zg: torch.Tensor
    Xg: torch.Tensor
    r_i: torch.Tensor
    l_i: torch.Tensor

    def to(self, dtype: torch.dtype) -> None:
        self.Zg = self.Zg.to(dtype)
        self.Xg = self.Xg.to(dtype)


class Variance:
    """
    The structured decomposition of V, and the entry points that solve it.

    Holds no numerical state of the current iterate: every evaluation goes
    through direct_solve() or capacitance(), which return a fresh Solve. Which
    of the two to call is the model's decision — Variance offers both and knows
    nothing of the SMW flag.

    `blocks` is empty and `embed` is None for models built through the low-level
    constructor, which has no design to decompose. `structured` then has no
    effect: V() returns None and the caller falls back to ZGZ' + R.
    """

    def __init__(self, blocks=None, embed=None):
        self.blocks = blocks or []
        self.embed = embed

    # ---- construction ------------------------------------------------

    @classmethod
    def from_designs(cls, designs, masks, device):
        """
        Build the per-effect blocks from the raw (pre-block_diag) designs.

        `designs` is a list of (M, c, L, comp), residual last. `masks` are the
        per-response boolean masks used to stack observations, so that F is the
        block diagonal of the per-response component values and lev follows the
        same stacking.
        """
        masks_t = [torch.as_tensor(m, dtype=torch.bool, device=device) for m in masks]
        blocks = []

        for i, (M, c, L, comp) in enumerate(designs):
            # the frozen K_obs must be built at the reference dtype: an explicit
            # step here rather than a side effect buried in a constructor
            comp.migrate(torch.double)

            M3 = torch.as_tensor(M, dtype=torch.double, device=device).reshape(len(M), c, L)
            F = torch.block_diag(*[M3[m].sum(-1) for m in masks_t])

            # the level carried by each row. An all-zero row has no level and
            # argmax returns 0 arbitrarily, which is harmless: its F row is zero
            # too, so the term it would contribute vanishes anyway.
            lev = M3.abs().sum(1).argmax(1)
            lev_obs = torch.cat([lev[m] for m in masks_t])

            k_obs = None
            if comp.right_hand in ("iid", "str"):
                K = comp.build_K()
                k_obs = K[lev_obs][:, lev_obs]

            blocks.append(Block(
                F           = F,
                lev         = lev_obs,
                k_obs       = k_obs,
                comp        = comp,
                is_residual = (i == len(designs) - 1),
            ))

        return cls(blocks=blocks)

    def embed_residual(self, Z, X, W, residual) -> None:
        """
        Attach the SMW residual embedding, when the Kronecker identities hold.

        Left as None otherwise — a dense-masked residual on unbalanced data, the
        multivariate case with missing observations notably. The caller reads
        `embed is None` as the operability rule that turns SMW off.
        """
        self.embed = None
        if Z is None:
            return
        if not (residual.Rtrick or Z.shape[0] == residual.d * residual.L):
            return

        d, L = residual.d, residual.L
        grid = W.argmax(1)
        Zg = Z.new_zeros(d * L, Z.shape[1])
        Zg[grid] = Z
        Xg = Z.new_zeros(d * L, X.shape[1])
        Xg[grid] = X

        self.embed = Embedding(grid=grid, Zg=Zg, Xg=Xg, r_i=grid // L, l_i=grid % L)

    def to(self, dtype: torch.dtype) -> None:
        """Follow the model's working dtype, for the frozen tensors only."""
        for blk in self.blocks:
            blk.to(dtype)
        if self.embed is not None:
            self.embed.to(dtype)

    @property
    def random_blocks(self) -> list:
        """Blocks in Z-column order, residual excluded."""
        return [b for b in self.blocks if not b.is_residual]

    @property
    def residual_block(self) -> Optional[Block]:
        for blk in self.blocks:
            if blk.is_residual:
                return blk
        return None

    # ---- solve entry points ------------------------------------------

    def direct_solve(self, X, r, dense_V: Callable, structured: bool) -> "DirectSolve":
        """
        Factor V on the direct path. `dense_V` is the model's fallback, called
        only when the structured forward is off.
        """
        return DirectSolve(self, X, r, dense_V, structured)

    def capacitance(self, X, Z, r, residual, dense_inv: Callable,
                    structured: bool) -> "Capacitance":
        """
        Factor C = Ginv + Z'R-inverse Z on the SMW path, V never formed.
        `dense_inv` is the model's varmeth_inv, used when the structured forward
        is off.
        """
        return Capacitance(self, X, Z, r, residual, dense_inv, structured)

    # ---- structured forward pieces -----------------------------------

    def V(self) -> Optional[torch.Tensor]:
        """
        Assemble V as the sum of the structured block terms. Returns None when
        there is no decomposition to exploit, letting the caller fall back.
        """
        if not self.blocks:
            return None
        V = None
        for blk in self.blocks:
            term = blk.term()
            V = term if V is None else V + term
        return V

    def Rinv_apply(self, residual) -> tuple[Callable, torch.Tensor]:
        """
        Return (apply, logdet_R), with `apply` the action of Rinv on a matrix
        already lifted to the (d·L, ·) space.

        Neither Rinv (n×n) nor the (d·L)² Kronecker block is formed: a diagonal
        residual reduces to an elementwise scale, a full one to a pair of
        contractions with Sinv and Kinv. Also called by the analytic backward on
        a full residual, which needs Rinv·Z even when the forward ran dense.
        """
        d, n_lev = residual.d, residual.L
        Sinv, logdet_S = residual.build_Sinv()
        Kinv, logdet_K = residual.build_Kinv()

        if residual.R_is_diagonal:
            sd = Sinv.diag()[:, None, None]
            kd = Kinv.diag()[None, :, None]

            def apply(Mg):
                return (Mg.reshape(d, n_lev, -1) * sd * kd).reshape(d * n_lev, -1)

            logdet_R = -torch.sum(
                torch.log(Sinv.diag()[self.embed.r_i] * Kinv.diag()[self.embed.l_i])
            )
        else:
            def apply(Mg):
                U = Mg.reshape(d, n_lev, -1)
                return torch.einsum('ij,jlm,kl->ikm', Sinv, U, Kinv).reshape(d * n_lev, -1)

            logdet_R = n_lev * logdet_S + d * logdet_K

        return apply, logdet_R

    def lift(self, M: torch.Tensor) -> torch.Tensor:
        """Scatter an (n, ·) matrix into the full (d·L, ·) space."""
        Mg = self.embed.Zg.new_zeros(self.embed.Zg.shape[0], M.shape[1])
        Mg[self.embed.grid] = M
        return Mg


class Solve:
    """
    What the two paths share: the three-term gradient matrix, and the beta
    grain read off it.

    Not an interface. `DirectSolve` and `Capacitance` each expose logdet_V,
    quad, k_reml, L and grains(), but nothing here declares or enforces that
    """

    def __init__(self, variance: Variance, X: torch.Tensor, r: torch.Tensor):
        self.variance = variance
        self.X = X
        self.r = r

        # V-inverse r, cached by gradient_matrix() for beta_grain()
        self._u: Optional[torch.Tensor] = None

    @staticmethod
    def _three_terms(P, u, W, Mt):
        """
        The gradient matrix, in whichever space the caller works:

            P - uu' - W Mt W'.

        Direct path: P = V-inverse, u = V-inverse r, W = V-inverse X,
        Mt = (X'V-inverse X)-inverse. SMW path: the same, each factor
        left-multiplied by Z'.
        """
        return P - u @ u.T - W @ Mt @ W.T

    def beta_grain(self) -> torch.Tensor:
        """
        dl/dbeta = -2 X'V-inverse r. beta is the only parameter not reachable
        from the variance blocks. Requires gradient_matrix() to have run.
        """
        if self._u is None:
            raise RuntimeError("beta_grain() called before gradient_matrix()")
        return -2.0 * (self.X.T @ self._u)


class DirectSolve(Solve):
    """
    Direct path: one Cholesky of the full V.

    The structured forward only changes how V is assembled — from the block
    terms rather than from ZGZ' + R — so both routes hit the same factorization
    and the same three terms, agreeing to roundoff.
    """

    def __init__(self, variance, X, r, dense_V: Callable, structured: bool):
        super().__init__(variance, X, r)

        V = variance.V() if structured else None
        if V is None:
            V = dense_V()

        self.L = torch.linalg.cholesky(V)
        M = torch.linalg.solve_triangular(self.L, r, upper=False)

        self.logdet_V = 2.0 * torch.sum(torch.log(torch.diag(self.L)))
        self.quad = (M.T @ M).squeeze()
        self.k_reml = torch.logdet(X.T @ torch.cholesky_solve(X, self.L))

    def gradient_matrix(self) -> torch.Tensor:
        """
        A = V⁻¹ − V⁻¹rr'V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹, from the stored factor.

        V⁻¹ = L⁻ᵀL⁻¹. torch.cholesky_inverse is pathologically slow in CPU
        double, so the triangular inverse goes through BLAS trsm and one
        symmetric matmul, and u and Wp come from L⁻¹ rather than from two extra
        cholesky_solve calls.
        """
        L = self.L
        I = torch.eye(L.shape[0], dtype=L.dtype, device=L.device)
        Li = torch.linalg.solve_triangular(L, I, upper=False)

        Vi = Li.T @ Li
        self._u = Li.T @ (Li @ self.r)
        Wp = Li.T @ (Li @ self.X)
        Mt = torch.linalg.inv(self.X.T @ Wp)

        return self._three_terms(Vi, self._u, Wp, Mt)

    def grains(self) -> Iterator[tuple]:
        A = self.gradient_matrix()
        for blk in self.variance.blocks:
            yield from blk.grain(A)


class Capacitance(Solve):
    """
    SMW path: one Cholesky of C = Ginv + Z'R⁻¹Z, V never formed.

    The structured forward only changes how Rinv is applied — as a Kronecker
    multiply on the lifted space rather than as a dense n×n inverse — so both
    routes assemble the same terms in the same order.
    """

    def __init__(self, variance, X, Z, r, residual, dense_inv: Callable,
                 structured: bool):
        super().__init__(variance, X, r)
        self.Z = Z
        self.residual = residual

        if structured:
            embed = variance.embed
            apply, logdet_R = variance.Rinv_apply(residual)

            rg = variance.lift(r)
            applyZ = apply(embed.Zg)
            applyR = apply(rg)
            applyX = apply(embed.Xg)

            self.ZtRiZ = embed.Zg.T @ applyZ
            self.Rir = applyR[embed.grid]
            self.ZtRir = embed.Zg.T @ applyR
            self.RiX = applyX[embed.grid]
            self.ZtRiX = embed.Zg.T @ applyX
            self.RinvZ = applyZ[embed.grid]
            self.P_full = applyZ

            # random effects only: the residual Rinv is already applied above
            inv_logdets = [b.comp.varmeth_inv()() for b in variance.random_blocks]
            Ginv = torch.block_diag(*[gi for gi, _ in inv_logdets])
            logdet_G = sum(ld for _, ld in inv_logdets)

        else:
            Ginv, Rinv, logdet_G, logdet_R = dense_inv()
            ZtRinv = Z.T @ Rinv
            self.ZtRiZ = ZtRinv @ Z
            self.RinvZ = ZtRinv.T
            self.Rir = Rinv @ r
            self.ZtRir = Z.T @ self.Rir
            self.RiX = Rinv @ X
            self.ZtRiX = Z.T @ self.RiX
            self.P_full = None

        C = Ginv + self.ZtRiZ
        self.L = torch.linalg.cholesky(C)
        logdet_C = 2.0 * torch.sum(torch.log(torch.diagonal(self.L)))

        self.logdet_V = logdet_R + logdet_G + logdet_C
        self.quad = (r.T @ self.Rir).squeeze() \
            - (self.ZtRir.T @ torch.cholesky_solve(self.ZtRir, self.L)).squeeze()

        self.XtViX = X.T @ self.RiX \
            - self.ZtRiX.T @ torch.cholesky_solve(self.ZtRiX, self.L)
        self.k_reml = torch.logdet(self.XtViX)
        self.Lx = torch.linalg.cholesky(self.XtViX)

        self._ViX: Optional[torch.Tensor] = None

    def _project(self, ZtRiM: torch.Tensor) -> torch.Tensor:
        """Z'V⁻¹M = Z'R⁻¹M − (Z'R⁻¹Z)C⁻¹(Z'R⁻¹M), the Woodbury projection."""
        return ZtRiM - self.ZtRiZ @ torch.cholesky_solve(ZtRiM, self.L)

    def gradient_matrix(self) -> torch.Tensor:
        """
        M = Z'V⁻¹Z − Z'V⁻¹rr'V⁻¹Z − Z'V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹Z.

        The same three terms as the direct path, projected on the q×q
        capacitance space. V is never formed; u and ViX are cached for the
        residual grain and for beta_grain().
        """
        self._u = self.Rir - self.RinvZ @ torch.cholesky_solve(self.ZtRir, self.L)
        self._ViX = self.RiX - self.RinvZ @ torch.cholesky_solve(self.ZtRiX, self.L)

        ZVir = self._project(self.ZtRir)
        ZViX = self._project(self.ZtRiX)
        ZViZ = self._project(self.ZtRiZ)

        return self._three_terms(
            ZViZ, ZVir, ZViX, torch.cholesky_inverse(self.Lx)
        )

    def grains(self) -> Iterator[tuple]:
        M = self.gradient_matrix()
        yield from self._random_grains(M)
        yield from self._residual_grain()

    def _random_grains(self, M: torch.Tensor) -> Iterator[tuple]:
        """
        With the block of effect e reshaped as M4 (d, L, d, L):

            grain_S[i,j] = sum_lm M4[i,l,j,m] K[l,m],
            grain_K[l,m] = sum_ij S[i,j] M4[i,l,j,m].
        """
        off = 0
        for blk in self.variance.random_blocks:
            rnd = blk.comp
            qe = rnd.d * rnd.L
            Me = M[off:off + qe, off:off + qe]
            off += qe

            S = rnd.build_S_full()
            K = rnd.build_K()
            M4 = Me.reshape(rnd.k * rnd.c, rnd.L, rnd.k * rnd.c, rnd.L)

            yield S, torch.einsum('iljm,lm->ij', M4, K.detach()).detach()

            if rnd.right_hand not in ("iid", "str"):
                yield K, torch.einsum('ij,iljm->lm', S.detach(), M4).detach()

    def _residual_grain(self) -> Iterator[tuple]:
        """
        The residual grain, in whichever of the two regimes applies.

        Diagonal R, with A_ii the diagonal of the direct-path gradient matrix:
            grain_S[r,r] = sum_{i in r} A_ii K[l_i,l_i],
            grain_K[l,l] = sum_i A_ii S[r_i,r_i]   (trainable K only).

        Full R, over the response blocks of P = R-inverse Z and Y = V-inverse X:
            grain_S[r,s] = Sinv[r,s]*L - tr(C-inv P_r'P_s) - u_r'u_s
                           - tr((X'V-inv X)-inv Y_r'Y_s).

        Pairs against build_S_full(), not build_S(): the forward inverts S with
        the jitter applied, so the grain must meet the same graph or the
        relative-jitter contribution to dS/dtheta is silently dropped.
        """
        resid = self.residual
        embed = self.variance.embed
        d, n_lev = resid.d, resid.L

        Sinv, _ = resid.build_Sinv()
        Kinv, _ = resid.build_Kinv()
        S_full = resid.build_S_full()

        if resid.R_is_diagonal:
            ri, li = embed.r_i, embed.l_i
            w = Sinv[ri, ri] * Kinv[li, li]

            Iq = torch.eye(self.Z.shape[1], dtype=self.L.dtype, device=self.L.device)
            ZLi = self.Z @ torch.linalg.solve_triangular(self.L.T, Iq, upper=True)
            diag_Vi = w - w * w * (ZLi * ZLi).sum(1)

            Ip = torch.eye(self.X.shape[1], dtype=self.L.dtype, device=self.L.device)
            Tx = self._ViX @ torch.linalg.solve_triangular(self.Lx.T, Ip, upper=True)

            A_ii = (diag_Vi - (self._u * self._u).flatten() - (Tx * Tx).sum(1)).detach()

            grain_S = torch.zeros(d, d, dtype=A_ii.dtype, device=A_ii.device)
            grain_S.index_put_((ri, ri), A_ii / Kinv.detach()[li, li], accumulate=True)
            yield S_full, grain_S

            if resid.right_hand == "het":
                grain_K = torch.zeros(n_lev, n_lev, dtype=A_ii.dtype, device=A_ii.device)
                grain_K.index_put_((li, li), A_ii * S_full.detach()[ri, ri], accumulate=True)
                yield resid.build_K(), grain_K

            return

        # the lifted R-inverse Z is a structured-forward by-product; rebuild it
        # when the forward ran dense. Only this branch needs it.
        P = self.P_full
        if P is None:
            apply, _ = self.variance.Rinv_apply(resid)
            P = apply(embed.Zg)

        ug = P.new_zeros(d * n_lev)
        ug[embed.grid] = self._u.flatten()
        Yg = self.variance.lift(self._ViX)

        B = P @ torch.cholesky_inverse(self.L) @ P.T
        F = Yg @ torch.cholesky_inverse(self.Lx) @ Yg.T

        grain_S = torch.zeros(d, d, dtype=self.L.dtype, device=self.L.device)
        for a in range(d):
            ba = slice(a * n_lev, (a + 1) * n_lev)
            for b in range(d):
                bb = slice(b * n_lev, (b + 1) * n_lev)
                grain_S[a, b] = (Sinv[a, b] * n_lev - B[ba, bb].trace()
                                 - (ug[ba] @ ug[bb]) - F[ba, bb].trace())

        yield S_full, grain_S.detach()