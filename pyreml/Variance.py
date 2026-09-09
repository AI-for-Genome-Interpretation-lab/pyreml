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
    - comp: the Random or Residual component owning the variance parameters
    - k_is_constant: K carries no trainable parameter, so K_obs can be cached
    - k_is_identity: K == I, so K_obs is never materialized at all
    """
    F: torch.Tensor
    lev: torch.Tensor
    comp: object
    is_residual: bool = False
    k_is_constant: bool = False
    k_is_identity: bool = False
    k_is_diagonal: bool = False
    _k_obs: Optional[torch.Tensor] = None

    def K_obs(self, scratch: Optional[dict] = None) -> Optional[torch.Tensor]:
        """
        K restricted to the observed stacking, or None when K is the identity.

        Built on first call, not at construction: only the direct path reads it,
        and an n x n gather must not be paid before the model has chosen between
        the direct and the SMW path. `scratch` is a per-solve dict letting
        term() and grain() share one gather within a single evaluation.

        Returned attached: term() builds V from it and needs the graph when the
        analytic backward is off. grain() detaches at its own use site.
        """
        if self.k_is_identity:
            return None
        if self._k_obs is not None:
            return self._k_obs
        if scratch is not None and id(self) in scratch:
            return scratch[id(self)]

        if self.k_is_diagonal:
            # K = diag(d): K_obs[i,j] = d[lev_i] when the levels match, 0 otherwise.
            # The L x L matrix is never formed.
            d = self.comp.build_K_diag()
            K_obs = (self.lev[:, None] == self.lev[None, :]) * d[self.lev][:, None]
        else:
            K_obs = self.comp.build_K()[self.lev][:, self.lev]

        if self.k_is_constant:
            self._k_obs = K_obs
        elif scratch is not None:
            scratch[id(self)] = K_obs
        return K_obs

    def term(self, scratch: Optional[dict] = None) -> torch.Tensor:
        """This block's contribution to V: (F S F') ⊙ K_obs."""
        S = self.comp.build_S_full()
        if self.k_is_identity:
            # K = I: only the diagonal of F S F' survives the Hadamard product,
            # so the n x n product is skipped as well as the n x n kernel
            return torch.diag_embed(((self.F @ S) * self.F).sum(1))
        return (self.F @ S @ self.F.T) * self.K_obs(scratch)

    def grain(self, A: torch.Tensor, scratch: Optional[dict] = None) -> list:
        """
        Gradient constants wrt this block's variance parameters, at the current
        point, from the model-level gradient matrix A:

            grain_S = F'(A ⊙ K_obs)F                 (dl/dS)
            grain_K = (A ⊙ F S F')[lev][:, lev]      (dl/dK, trainable K only)

        Returns the (tensor, grain) pairs whose inner products form the ghost
        loss: the tensor carries the parameter graph, the grain is a constant.
        Pairing against build_S_full reproduces dl/dtheta exactly, jitter
        included, since the relative-jitter term lives in that same graph.

        A diagonal K is paired on its diagonal: only grain_K[l,l] contributes,
        so neither K nor grain_K is formed as an L x L matrix.
        """
        S = self.comp.build_S_full()

        if self.k_is_identity:
            # A ⊙ I = diag(A), so grain_S = F' diag(A) F
            a = torch.diagonal(A).detach()
            pairs = [(S, ((self.F * a[:, None]).T @ self.F).detach())]
        else:
            K_obs = self.K_obs(scratch)
            pairs = [(S, (self.F.T @ (A * K_obs.detach()) @ self.F).detach())]

        if self.k_is_constant:
            return pairs

        K_pair = self.comp.build_K_diag() if self.k_is_diagonal else self.comp.build_K()

        M = (A * (self.F @ S.detach() @ self.F.T)).detach()
        n_lev = self.comp.L
        # scatter, not gather: dl/dK[l,m] sums the observation-level entries
        # over every pair (i,j) landing on levels (l,m). A gather indexed by
        # lev reads an n×n matrix at level positions, which only coincides
        # when L == n and lev is the identity — true for the ungridded hands,
        # out of bounds for the gridded AR structures where L > n.
        rows = M.new_zeros(n_lev, M.shape[1]).index_add_(0, self.lev, M)

        if self.k_is_diagonal:
            # only the diagonal of grain_K is paired, so the second scatter
            # collapses to one gather: grain[l] = Σ_{i,j ∈ l} M[i,j]
            vals = rows.T.gather(1, self.lev[:, None]).squeeze(1)
            grain_K = M.new_zeros(n_lev).index_add_(0, self.lev, vals)
            pairs.append((K_pair, grain_K))
        else:
            grain_K = M.new_zeros(n_lev, n_lev).index_add_(0, self.lev, rows.T).T
            pairs.append((K_pair, grain_K))

        return pairs

    def to(self, dtype: torch.dtype) -> None:
        """Cast the frozen tensors in place, following the model's working dtype."""
        self.F = self.F.to(dtype)
        # dropped rather than cast: rebuilt lazily at the new dtype, and only if
        # the direct path actually asks for it
        self._k_obs = None


@dataclass
class Embedding:
    """
    Residual structure of the SMW path.

    Zg and Xg lift the incidence and the design into the full (response × level)
    space, so that applying Rinv becomes a Kronecker multiply (Sinv ⊗ Kinv) and
    Rinv (n×n) is never formed. Exists only when the Kronecker identities hold:
    a Rtrick residual (fully diagonal) or balanced data (no missing cell). Its
    absence is what turns SMW off for good, in from_dataframe.

    The Kronecker multiply is itself skipped on a fully diagonal residual: Rinv
    is a per-observation scale then, `r_i`/`l_i` are its only operands and the
    whole incidence product runs through Variance.ZtM/Zv/ZtWZ. Zg is None in
    that regime (the from_dataframe constructor does not materialize Z), while
    the non-diagonal structured path still needs it.

    Under an identity selector the lift is a no-op: Zg and Xg alias the model's
    own Z and X instead of being scattered into fresh (d·L, ·) buffers, and the
    read-back at the observed cells is skipped as well. `is_identity` records
    that, so migrate() can re-establish the alias after a dtype change.
    """
    grid: torch.Tensor
    Zg: torch.Tensor
    Xg: torch.Tensor
    r_i: torch.Tensor
    l_i: torch.Tensor
    is_identity: bool = False

    def to(self, dtype: torch.dtype) -> None:
        # aliased operands are cast by the model, not here: casting them would
        # break the alias and hold a second copy of Z at the working dtype.
        # The model re-points them in migrate() right after casting Z and X.
        if self.is_identity:
            return
        if self.Zg is not None:
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
        masks_t = [torch.tensor(m, dtype=torch.bool, device=device) for m in masks]
        blocks = []

        for i, (M, c, L, comp) in enumerate(designs):
            # the sanctuarized constants are derived once, in double, before any
            # working dtype is applied
            comp.migrate(torch.double)
            is_residual = (i == len(designs) - 1)

            if is_residual:
                # the residual design is a selector: every row carries a single
                # unit value, so F is a response indicator and lev is the grid
                lev = torch.as_tensor(M, dtype=torch.long, device=device)
                F = torch.block_diag(*[
                    torch.ones(int(m.sum()), 1, dtype=torch.double, device=device)
                    for m in masks_t
                ])
            else:
                if hasattr(comp, "lev_base"):
                    # canonical factored incidence (make_Z): (F_base, lev_base)
                    # already carries the per-column values and the level each
                    # row loads, so the n×c·L working matrix is never moved to
                    # the device. lev_base follows Z through the str/dist and
                    # ar relayouts, so it lines up with the re-laid columns.
                    F = torch.block_diag(*[
                        torch.as_tensor(comp.F_base[m], dtype=torch.double, device=device)
                        for m in masks
                    ])
                    lev = torch.as_tensor(comp.lev_base, dtype=torch.long, device=device)
                else:
                    M3 = torch.as_tensor(M, dtype=torch.double, device=device).reshape(len(M), c, L)
                    F = torch.block_diag(*[M3[m].sum(-1) for m in masks_t])

                    # the level carried by each row. An all-zero row has no level and
                    # argmax returns 0 arbitrarily, which is harmless: its F row is zero
                    # too, so the term it would contribute vanishes anyway.
                    lev = M3.abs().sum(1).argmax(1)

            lev_obs = torch.cat([lev[m] for m in masks_t])

            # K = I gives K_obs[i,j] = δ(lev_i, lev_j), which is the identity
            # only when no level is loaded twice. Several responses share the
            # levels of an effect, so lev_obs decides, not right_hand.
            k_iid = comp.right_hand == "iid"
            k_is_identity = k_iid and int(lev_obs.unique().numel()) == int(lev_obs.numel())

            # K_obs is not built here: only the direct path reads it, and the
            # path is not chosen yet. The block records what it may cache and
            # gathers on first use.
            blocks.append(Block(
                F             = F,
                lev           = lev_obs,
                comp          = comp,
                is_residual   = is_residual,
                k_is_constant = comp.right_hand in ("iid", "str"),
                k_is_identity = k_is_identity,
                k_is_diagonal = comp.K_is_diagonal,
            ))

        return cls(blocks=blocks)

    def embed_residual(self, Z, X, grid, residual) -> None:
        """
        Attach the SMW residual embedding, when the Kronecker identities hold.

        Left as None otherwise — a dense-masked residual on unbalanced data, the
        multivariate case with missing observations notably. The caller reads
        `embed is None` as the operability rule that turns SMW off.
        """
        self.embed = None
        # no random effect: the embedding has nothing to serve, and its absence
        # is exactly what keeps SMW off (see the resolution in from_dataframe).
        # Random blocks require it even when the incidence is factored (Z is
        # None): the diagonal path reads r_i/l_i off it, with Zg left None in
        # that regime.
        if Z is None and not self.random_blocks:
            return
        if not (residual.Rtrick or grid.numel() == residual.d * residual.L):
            return

        d, L = residual.d, residual.L

        if residual.W_is_identity:
            # the lift is a no-op: alias rather than scatter. Grid is arange(n)
            # with n == d·L, so Zg[grid] = Z is the identity mapping, and the
            # two buffers would be exact copies of the model's Z and X. With a
            # factored incidence (Z is None) the diagonal Rinv needs neither
            # Zg nor Xg: every product runs through Variance.ZtM/Zv/ZtWZ on the
            # observed rows, so they are never materialized.
            Zg, Xg = Z, X
        else:
            Zg = Z.new_zeros(d * L, Z.shape[1]) if Z is not None else None
            if Zg is not None:
                Zg[grid] = Z
            Xg = X.new_zeros(d * L, X.shape[1])
            Xg[grid] = X

        self.embed = Embedding(
            grid=grid, Zg=Zg, Xg=Xg,
            r_i=grid // L, l_i=grid % L,
            is_identity=residual.W_is_identity,
        )

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

    # ---- factored incidence facing -----------------------------------

    def _z_layout(self) -> tuple[list[int], int]:
        """
        Column offsets of the per-effect blocks and the total column count q.

        The same element-outer / level-inner convention as Random.make_Z: the
        e-th block spans columns [off_e, off_e + d_e·L_e) and observation i
        loads its values on columns off_e + j·L_e + lev_e[i].
        """
        offsets, q = [], 0
        for blk in self.random_blocks:
            offsets.append(q)
            q += blk.comp.d * blk.comp.L
        return offsets, q

    def ZtM(self, M: torch.Tensor) -> torch.Tensor:
        """Z' M without forming Z, M of shape (n, ·)."""
        offsets, q = self._z_layout()
        out = torch.zeros(q, M.shape[1], dtype=M.dtype, device=M.device)
        for blk, off in zip(self.random_blocks, offsets):
            F, lev, L, d = blk.F, blk.lev, blk.comp.L, blk.comp.d
            for j in range(d):
                out.index_add_(0, off + j * L + lev, F[:, j, None] * M)
        return out

    def Zv(self, v: torch.Tensor) -> torch.Tensor:
        """Z v without forming Z, v of shape (q, ·)."""
        offsets, _ = self._z_layout()
        n = self.random_blocks[0].lev.numel()
        out = torch.zeros(n, v.shape[1], dtype=v.dtype, device=v.device)
        for blk, off in zip(self.random_blocks, offsets):
            F, lev, L, d = blk.F, blk.lev, blk.comp.L, blk.comp.d
            for j in range(d):
                out += F[:, j, None] * v[off + j * L + lev]
        return out

    def ZtWZ(self, w: torch.Tensor) -> torch.Tensor:
        """
        Z' diag(w) Z without forming Z, w a (n,) weight per observation.

        Every observation contributes w_i z_i z_i' with z_i the incidence row,
        so its weight lands on the (q, q) entry indexed by its level cells in
        the two blocks involved. When w is the diagonal of Rinv this is Z'Rinv Z
        on the structured diagonal path.
        """
        offsets, q = self._z_layout()
        out = torch.zeros(q * q, dtype=w.dtype, device=w.device)
        for a, off_a in zip(self.random_blocks, offsets):
            for b, off_b in zip(self.random_blocks, offsets):
                for ja in range(a.comp.d):
                    ra = off_a + ja * a.comp.L + a.lev
                    for jb in range(b.comp.d):
                        rb = off_b + jb * b.comp.L + b.lev
                        out.index_add_(
                            0, ra * q + rb, w * a.F[:, ja] * b.F[:, jb]
                        )
        return out.reshape(q, q)

    def _diag_ZCinvZ(self, Cinv: torch.Tensor) -> torch.Tensor:
        """diag(Z Cinv Z'), taken per observation on the factored incidence.

        The diagonal element of observation i reads Cinv at the column pairs
        its row supports, off_e + j·L_e + lev_e[i], so it is gathered from the
        capacitance inverse rather than formed through an n×q ZLii product.
        """
        offsets, _ = self._z_layout()
        n = self.random_blocks[0].lev.numel()
        acc = torch.zeros(n, dtype=Cinv.dtype, device=Cinv.device)
        for a, off_a in zip(self.random_blocks, offsets):
            for b, off_b in zip(self.random_blocks, offsets):
                for ja in range(a.comp.d):
                    ra = off_a + ja * a.comp.L + a.lev
                    for jb in range(b.comp.d):
                        rb = off_b + jb * b.comp.L + b.lev
                        acc += a.F[:, ja] * b.F[:, jb] * Cinv[ra, rb]
        return acc

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

    def V(self, scratch: Optional[dict] = None) -> Optional[torch.Tensor]:
        """
        Assemble V as the sum of the structured block terms. Returns None when
        there is no decomposition to exploit, letting the caller fall back.
        """
        if not self.blocks:
            return None
        V = None
        for blk in self.blocks:
            term = blk.term(scratch)
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

        if residual.K_is_diagonal:
            # K inverse is diagonal: keep only its diagonal, never form the
            # L×L Kinv. Reachable either with a diagonal R (scale on the
            # levels) or, when K is diagonal but S is not, with a single
            # contraction over the response dimension.
            kd, logdet_K = residual.build_Kinv_diag()
        else:
            Kinv, logdet_K = residual.build_Kinv()

        if residual.R_is_diagonal:
            sd = Sinv.diag()[:, None, None]
            kd = kd[None, :, None]

            def apply(Mg):
                return (Mg.reshape(d, n_lev, -1) * sd * kd).reshape(d * n_lev, -1)

            logdet_R = -torch.sum(
                torch.log(Sinv.diag()[self.embed.r_i] * kd.flatten()[self.embed.l_i])
            )
        elif residual.K_is_diagonal:
            def apply(Mg):
                U = Mg.reshape(d, n_lev, -1)
                return torch.einsum('ij,jlm->ilm', Sinv, U * kd[None, :, None]).reshape(d * n_lev, -1)

            logdet_R = n_lev * logdet_S + d * logdet_K
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

        # per-evaluation scratch: term() and grain() gather the same K_obs for a
        # trainable K, so the second call reads the first one's result. Dropped
        # with the Solve, so it can never go stale across iterations.
        self._scratch: dict = {}

        V = variance.V(self._scratch) if structured else None
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
            yield from blk.grain(A, self._scratch)


class Capacitance(Solve):
    """
    SMW path: one Cholesky of C = Ginv + Z'R⁻¹Z, V never formed.

    The structured forward only changes how Rinv is applied — a Kronecker
    multiply on the lifted space for a full R, a per-observation scale on a
    fully diagonal R — so every route assembles the same terms in the same
    order. On the diagonal route the products run through the factored
    incidence facing and the dense Z never exists.
    """

    def __init__(self, variance, X, Z, r, residual, dense_inv: Callable,
                 structured: bool):
        super().__init__(variance, X, r)
        self.Z = Z
        self.residual = residual
        self._w: Optional[torch.Tensor] = None
        self.RinvZ: Optional[torch.Tensor] = None

        if structured:
            embed = variance.embed

            if residual.R_is_diagonal:
                # fully diagonal R: Rinv is an elementwise scale on the observed
                # cells, and every Z product runs through the per-block
                # incidence facing. Z itself may never be materialized.
                Sinv, _ = residual.build_Sinv()
                kd, _ = residual.build_Kinv_diag()
                w = (Sinv[embed.r_i, embed.r_i] * kd[embed.l_i]).reshape(-1, 1)

                self.ZtRiZ = variance.ZtWZ(w.reshape(-1))
                self.ZtRir = variance.ZtM(w * r)
                self.ZtRiX = variance.ZtM(w * X)
                self.Rir = w * r
                self.RiX = w * X
                self.P_full = None
                self._w = w

                logdet_R = -torch.sum(torch.log(w.reshape(-1)))
            else:
                apply, logdet_R = variance.Rinv_apply(residual)

                rg = variance.lift(r)
                applyZ = apply(embed.Zg)
                applyR = apply(rg)
                applyX = apply(embed.Xg)

                self.ZtRiZ = embed.Zg.T @ applyZ
                self.ZtRir = embed.Zg.T @ applyR
                self.ZtRiX = embed.Zg.T @ applyX
                self.P_full = applyZ

                if embed.is_identity:
                    # the read-back is the identity too: alias instead of gathering
                    self.Rir, self.RiX, self.RinvZ = applyR, applyX, applyZ
                else:
                    self.Rir = applyR[embed.grid]
                    self.RiX = applyX[embed.grid]
                    self.RinvZ = applyZ[embed.grid]

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
        if self._w is not None:
            # diagonal R: Rinv Z x = w ⊙ (Z x), applied on the factored incidence
            self._u = self.Rir - self._w * self.variance.Zv(
                torch.cholesky_solve(self.ZtRir, self.L)
            )
            self._ViX = self.RiX - self._w * self.variance.Zv(
                torch.cholesky_solve(self.ZtRiX, self.L)
            )
        else:
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
        S_full = resid.build_S_full()

        if resid.R_is_diagonal:
            # K is diagonal in this regime (right_hand iid/het): only its
            # inverse diagonal is ever read, so the L×L Kinv is never formed.
            kd, _ = resid.build_Kinv_diag()
            ri, li = embed.r_i, embed.l_i
            w = Sinv[ri, ri] * kd[li]

            # diag(Z C-inverse Z') without the n×q Z L^-T product: the row of
            # observation i supports columns off_e + j·L_e + lev_e[i], so its
            # diagonal element gathers Cinv at those column pairs — L L' = C is
            # the capacitance factor this solve was built on.
            Cinv = torch.cholesky_inverse(self.L)
            diag_Vi = w - w * w * self.variance._diag_ZCinvZ(Cinv)

            Ip = torch.eye(self.X.shape[1], dtype=self.L.dtype, device=self.L.device)
            Tx = self._ViX @ torch.linalg.solve_triangular(self.Lx.T, Ip, upper=True)

            A_ii = (diag_Vi - (self._u * self._u).flatten() - (Tx * Tx).sum(1)).detach()

            grain_S = torch.zeros(d, d, dtype=A_ii.dtype, device=A_ii.device)
            grain_S.index_put_((ri, ri), A_ii / kd.detach()[li], accumulate=True)
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