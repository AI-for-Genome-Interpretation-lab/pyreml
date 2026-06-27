# %%
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from scipy.optimize import minimize_scalar, minimize

from pyreml import larix as df

rrBLUP = importr("rrBLUP")

df = df[df["year"] == 2000].copy()
df["ID"] = np.arange(len(df))

y = df["height"].to_numpy()
n = len(y)
Xmat = np.ones((n, 1))

# hold-out re-predicted as "new" levels (same blocks as the original campaign)
df_pred = df[df["BLOC"].isin([f"B{i}" for i in range(13, 17)])].copy()
m = len(df_pred)


# --------------------------------------------------------------------------- #
# Kernels — each maps a set of positions P (M x axes) and a rate vector to K.
# This is the single source of truth shared by every reference: changing the
# rate count or the metric is all that distinguishes the four references.
# --------------------------------------------------------------------------- #
def _pairwise_dist(P):
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def euclidean_K(P, rhos):
    # exp(-rho * ||dx||_2): circular contours. In 1D this is the exponential
    # decay shared by eucl / ar_iso / ar_ani.
    return np.exp(-rhos[0] * _pairwise_dist(P))


def manhattan_iso_K(P, rhos):
    # exp(-rho * sum_a |dx_a|): one shared rate, L1 / diamond contours.
    L1 = np.abs(P[:, None, :] - P[None, :, :]).sum(axis=-1)
    return np.exp(-rhos[0] * L1)


def separable_ani_K(P, rhos):
    # exp(-sum_a rho_a |dx_a|): one rate per axis, anisotropic.
    A = np.abs(P[:, None, :] - P[None, :, :])          # (M, M, axes)
    return np.exp(-(A * np.asarray(rhos)).sum(axis=-1))


# --------------------------------------------------------------------------- #
# Level collapse — repeated coordinates (the 1D case along X) share a single
# level, carried by an incidence Z; distinct coordinates (the 2D grid) give
# Z = I in row order. Levels are kept in FIRST-OCCURRENCE order, which for the
# 2D grid coincides with the ID/row order of the original spatial.json.
#
# /!\ This order is the contract with the model: the design's `self.index` for
#     a coordinate effect must enumerate levels the same way, or `blup` / `pev`
#     will compare across mismatched orderings.
# --------------------------------------------------------------------------- #
def collapse(P):
    keys = pd.Series(list(map(tuple, P)))
    codes, uniques = pd.factorize(keys)                # first-occurrence order
    P_lev = np.array([list(t) for t in uniques], dtype=float)
    Z = np.zeros((len(P), len(uniques)))
    Z[np.arange(len(P)), codes] = 1.0
    return P_lev, Z


# --------------------------------------------------------------------------- #
# rrBLUP wrappers
# --------------------------------------------------------------------------- #
def _mixed_solve(Z, K, X=Xmat, se=False):
    with localconverter(ro.default_converter + numpy2ri.converter):
        y_r = ro.conversion.py2rpy(y)
        Z_r = ro.conversion.py2rpy(Z)
        K_r = ro.conversion.py2rpy(K)
        X_r = ro.conversion.py2rpy(X)
    return rrBLUP.mixed_solve(y=y_r, Z=Z_r, K=K_r, X=X_r, method="REML", SE=se)


def _profile_rho(kernel, P_lev, Z, n_rho, bounds=(1e-6, 10.0)):
    """External REML profiling of the decay rate(s) on the rrBLUP log-likelihood."""
    def neg_LL(rho_vec):
        rho_vec = np.atleast_1d(rho_vec).astype(float)
        if np.any(rho_vec <= 0):
            return 1e10
        fit = _mixed_solve(Z, kernel(P_lev, rho_vec))
        return -float(fit.rx2("LL")[0])

    if n_rho == 1:
        res = minimize_scalar(lambda r: neg_LL([r]), bounds=bounds, method="bounded")
        return [float(res.x)]

    res = minimize(neg_LL, x0=np.ones(n_rho), method="Nelder-Mead")
    return [float(v) for v in res.x]


# --------------------------------------------------------------------------- #
# One call generates a full reference: profile -> fit (SE) -> kriging hold-out,
# and dumps spat_{name}.json + spat_{name}_pred.json. Returns a dict with
# everything the prediction map needs (levels, incidence, kernel, rate).
# --------------------------------------------------------------------------- #
def make_reference(name, cols, kernel, n_rho):
    P_train = df[cols].to_numpy(dtype=float)
    P_lev, Z = collapse(P_train)
    L = len(P_lev)

    # --- profile the rate(s) -------------------------------------------------
    rho = _profile_rho(kernel, P_lev, Z, n_rho)
    K = kernel(P_lev, rho)

    # --- fit with SE for Vu / Ve / beta / BLUP / PEV -------------------------
    fit = _mixed_solve(Z, K, se=True)
    Vu = float(fit.rx2("Vu")[0])
    Ve = float(fit.rx2("Ve")[0])
    beta = float(np.asarray(fit.rx2("beta")).ravel()[0])
    beta_se = float(np.asarray(fit.rx2("beta.SE")).ravel()[0])
    blup = np.asarray(fit.rx2("u")).ravel().tolist()
    pev_diag = (np.asarray(fit.rx2("u.SE")).ravel() ** 2).tolist()

    with open(f"../data/spat_{name}.json", "w") as f:
        json.dump({
            "rho": rho,                       # list in every case
            "Vu": Vu,
            "Ve": Ve,
            "beta": beta,
            "eev_intercept": beta_se ** 2,
            "blup": blup,
            "pev_diag": pev_diag,
        }, f, indent=2)

    # --- kriging hold-out: m pred rows appended as new levels ----------------
    # Train levels stay collapsed (L); each pred row is its own new level (m),
    # exactly as the model's predict conditions L train levels -> m new ones.
    P_pred = df_pred[cols].to_numpy(dtype=float)
    P_full = np.vstack([P_lev, P_pred])
    K_full = kernel(P_full, rho)
    Z_aug = np.hstack([Z, np.zeros((n, m))])          # (n, L + m)

    fit2 = _mixed_solve(Z_aug, K_full)
    blup_pred = np.asarray(fit2.rx2("u")).ravel()[L:].tolist()

    with open(f"../data/spat_{name}_pred.json", "w") as f:
        json.dump({"n_train": L, "n_pred": m, "blup_pred": blup_pred}, f, indent=2)

    print(f"spat_{name}: rho={rho}  Vu={Vu:.4g}  Ve={Ve:.4g}  L={L}  m={m}")
    return dict(name=name, cols=cols, kernel=kernel, rho=rho, P_lev=P_lev, Z=Z, L=L)


# --------------------------------------------------------------------------- #
# Validation map: build the kernel over [train levels ; raster grid], pass it to
# rrBLUP with the observations, and krige the random effect over the whole field.
# The result is a raster of predicted BLUP; the observations are overlaid as
# black points. For the 1D reference the prediction depends on X only, so the
# raster shows vertical bands — an immediate visual check of the structure.
# --------------------------------------------------------------------------- #
def prediction_map(ref, res=60):
    cols, kernel, rho = ref["cols"], ref["kernel"], ref["rho"]
    P_lev, Z, L = ref["P_lev"], ref["Z"], ref["L"]

    xs, ys = df["X"].to_numpy(), df["Y"].to_numpy()
    gx = np.linspace(xs.min(), xs.max(), res)
    gy = np.linspace(ys.min(), ys.max(), res)
    GX, GY = np.meshgrid(gx, gy)                       # (res, res), 'xy'
    grid_xy = np.column_stack([GX.ravel(), GY.ravel()])  # (res*res, 2): X, Y

    sel = [["X", "Y"].index(c) for c in cols]          # ref's coordinate columns
    P_grid = grid_xy[:, sel]                            # (n_grid, axes)
    n_grid = P_grid.shape[0]

    # full kernel over train levels + grid, solved with the observations
    K_full = kernel(np.vstack([P_lev, P_grid]), rho)
    Z_aug = np.hstack([Z, np.zeros((n, n_grid))])      # grid carries no data
    fit = _mixed_solve(Z_aug, K_full)
    u_grid = np.asarray(fit.rx2("u")).ravel()[L:].reshape(res, res)

    plt.figure(figsize=(5, 4))
    vmax = np.abs(u_grid).max()
    plt.imshow(
        u_grid, origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(label="kriged BLUP")
    plt.scatter(xs, ys, c="black", s=4)
    plt.title(f"spat_{ref['name']}: prediction map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## `spat_dist` — Euclidean 2D
# Shared by `dist`, `str` and `eucl`(2D). Circular contours.

# %%
ref = make_reference("dist", ["X", "Y"], euclidean_K, n_rho=1)
prediction_map(ref)


# %% [markdown]
# ## `spat_1d` — 1D exponential (along X)
# Shared by `eucl`(1D), `ar_iso`(1D), `ar_ani`(1D): the documented 1D identity.
# Repeated X values collapse onto shared levels through the incidence Z.
# The prediction depends on X only -> vertical bands.

# %%
ref = make_reference("1d", ["X"], euclidean_K, n_rho=1)
prediction_map(ref)


# %% [markdown]
# ## `spat_iso2d` — separable AR, isotropic
# One shared rate over both axes: Manhattan (L1) decay, diamond contours.

# %%
ref = make_reference("iso2d", ["X", "Y"], manhattan_iso_K, n_rho=1)
prediction_map(ref)


# %% [markdown]
# ## `spat_ani2d` — separable AR, anisotropic
# One rate per axis: elliptical (axis-aligned) contours. Profiled in 2D.

# %%
ref = make_reference("ani2d", ["X", "Y"], separable_ani_K, n_rho=2)
prediction_map(ref)