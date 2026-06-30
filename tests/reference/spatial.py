# %% [markdown]
# # Spatial / decay-kernel references (rrBLUP)
#
# Produces the references consumed by `test_spatial.py`:
#
#     spat_dist    <-  dist, str, eucl     Euclidean kernel
#     spat_iso     <-  ar_iso              Manhattan (diamond) kernel
#     spat_ani     <-  ar_ani              anisotropic kernel
#
# Runs on a contrasted-but-close bloc subset (SUBSET_BLOCS) chosen so the AR
# kernels sit in an identifiable regime (rho * spacing moderate, neither flat
# nor diagonal).

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

from pyreml import larix as DF

rrBLUP = importr("rrBLUP")

# Subset of blocs: contrasted but geographically close, so the AR kernels are
# identifiable. MUST be kept identical to SUBSET_BLOCS / PRED_OFFSET / _grid /
# _holdout in test_spatial.py: the model and the reference must see the exact
# same rows in the exact same order.
SUBSET_BLOCS = ["B3", "B13"]

# Hold-out coordinate offset. The fit runs on the whole subset, so every observed
# coordinate is a training level. To exercise kriging at genuinely NEW levels the
# held-out rows are re-predicted at their coordinates shifted by this offset:
# off the integer grid, guaranteeing new levels rather than duplicates. The TEST
# applies the very same offset when it reconstructs the prediction coordinates.
PRED_OFFSET = 0.5


# %% [markdown]
# ## Bloc map
# Plots the full trial to pick contrasted-but-close blocs by eye.

# %%
fig, ax = plt.subplots(figsize=(12, 8))

for _, row in DF.iterrows():
    ax.text(row["X"], row["Y"], str(row["BLOC"]), ha="center", va="center")

ax.set_xlim(DF["X"].min(), DF["X"].max())
ax.set_ylim(DF["Y"].min(), DF["Y"].max())
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Position des blocs")
ax.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Shared primitives
# Kept as functions because repeating the rrBLUP plumbing per cell would be
# unreadable; everything statistical (profiling, fit, dump, kriging) is inlined
# in each reference cell instead.

# %%
def _grid(df):
    """Year-2000 grid restricted to SUBSET_BLOCS; one level per row, unique (X, Y)."""
    df = df.loc[(df["year"] == 2000) & df["BLOC"].isin(SUBSET_BLOCS)].copy()
    df["ID"] = np.arange(len(df))
    return df


def _holdout(data):
    """Kriging hold-out: every 3rd row, re-predicted as NEW levels via PRED_OFFSET.
    Deterministic and independent of the bloc choice, so it stays valid as the
    subset is tuned."""
    return data.iloc[::3]


def euclidean_K(P, rhos):
    # exp(-rho * ||dx||_2): circular contours.
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    return np.exp(-rhos[0] * D)


def manhattan_iso_K(P, rhos):
    # exp(-rho * sum_a |dx_a|): one shared rate, L1 / diamond contours.
    L1 = np.abs(P[:, None, :] - P[None, :, :]).sum(axis=-1)
    return np.exp(-rhos[0] * L1)


def separable_ani_K(P, rhos):
    # exp(-sum_a rho_a |dx_a|): one rate per axis, anisotropic.
    A = np.abs(P[:, None, :] - P[None, :, :])
    return np.exp(-(A * np.asarray(rhos)).sum(axis=-1))


def collapse(P):
    """Repeated coordinates share a single level (incidence Z); distinct
    coordinates give Z = I in row order. Levels in FIRST-OCCURRENCE order, the
    contract with the model's `self.index` for a coordinate effect."""
    keys = pd.Series(list(map(tuple, P)))
    codes, uniques = pd.factorize(keys)
    P_lev = np.array([list(t) for t in uniques], dtype=float)
    Z = np.zeros((len(P), len(uniques)))
    Z[np.arange(len(P)), codes] = 1.0
    return P_lev, Z


def _mixed_solve(Z, K, y, X, se=False):
    with localconverter(ro.default_converter + numpy2ri.converter):
        y_r = ro.conversion.py2rpy(y)
        Z_r = ro.conversion.py2rpy(Z)
        K_r = ro.conversion.py2rpy(K)
        X_r = ro.conversion.py2rpy(X)
    return rrBLUP.mixed_solve(y=y_r, Z=Z_r, K=K_r, X=X_r, method="REML", SE=se)


def prediction_map(name, data, y, X, n, cols, kernel, rho, P_lev, Z, L, res=60):
    """Krige the random effect over the whole field and overlay observations.
    Only the training levels P_lev drive the kernel; the rest is purely predicted."""
    field = _grid(DF)
    xs, ys = field["X"].to_numpy(), field["Y"].to_numpy()
    gx = np.linspace(xs.min(), xs.max(), res)
    gy = np.linspace(ys.min(), ys.max(), res)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([GX.ravel(), GY.ravel()])
    sel = [["X", "Y"].index(c) for c in cols]
    P_grid = grid_xy[:, sel]
    n_grid = P_grid.shape[0]

    K_full = kernel(np.vstack([P_lev, P_grid]), rho)
    Z_aug = np.hstack([Z, np.zeros((n, n_grid))])
    fit = _mixed_solve(Z_aug, K_full, y, X)
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
    plt.title(f"spat_{name}: prediction map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## `spat_dist` — Euclidean
# Shared by `dist`, `str` and `eucl`. Circular contours, single rate.

# %%
data = _grid(DF)
holdout = _holdout(data)
cols = ["X", "Y"]
kernel = euclidean_K

y = data["height"].to_numpy()
n = len(y)
X = np.ones((n, 1))

P_train = data[cols].to_numpy(dtype=float)
P_lev, Z = collapse(P_train)
L = len(P_lev)

# --- profile the single decay rate on the rrBLUP REML log-likelihood ---------
def neg_LL(r):
    if r <= 0:
        return 1e10
    fit = _mixed_solve(Z, kernel(P_lev, [r]), y, X)
    return -float(fit.rx2("LL")[0])

res = minimize_scalar(neg_LL, bounds=(1e-6, 10.0), method="bounded")
rho = [float(res.x)]

# --- fit with SE for Vu / Ve / beta / BLUP / PEV ----------------------------
K = kernel(P_lev, rho)
fit = _mixed_solve(Z, K, y, X, se=True)
Vu = float(fit.rx2("Vu")[0])
Ve = float(fit.rx2("Ve")[0])
beta = float(np.asarray(fit.rx2("beta")).ravel()[0])
beta_se = float(np.asarray(fit.rx2("beta.SE")).ravel()[0])
blup = np.asarray(fit.rx2("u")).ravel().tolist()
pev_diag = (np.asarray(fit.rx2("u.SE")).ravel() ** 2).tolist()

with open("../data/spat_dist.json", "w") as f:
    json.dump({
        "rho": rho, "Vu": Vu, "Ve": Ve, "beta": beta,
        "eev_intercept": beta_se ** 2, "blup": blup, "pev_diag": pev_diag,
    }, f, indent=2)

# --- kriging hold-out: m pred rows appended as NEW levels --------------------
P_pred = holdout[cols].to_numpy(dtype=float) + PRED_OFFSET
m = len(P_pred)
K_full = kernel(np.vstack([P_lev, P_pred]), rho)
Z_aug = np.hstack([Z, np.zeros((n, m))])
fit2 = _mixed_solve(Z_aug, K_full, y, X)
blup_pred = np.asarray(fit2.rx2("u")).ravel()[L:].tolist()

with open("../data/spat_dist_pred.json", "w") as f:
    json.dump({"n_train": L, "n_pred": m, "blup_pred": blup_pred}, f, indent=2)

print(f"spat_dist: rho={rho} Vu={Vu:.4g} Ve={Ve:.4g} L={L} m={m}")
prediction_map("dist", data, y, X, n, cols, kernel, rho, P_lev, Z, L)


# %% [markdown]
# ## `spat_iso` — separable AR, isotropic
# One shared rate over both axes: Manhattan (L1) decay, diamond contours.

# %%
data = _grid(DF)
holdout = _holdout(data)
cols = ["X", "Y"]
kernel = manhattan_iso_K

y = data["height"].to_numpy()
n = len(y)
X = np.ones((n, 1))

P_train = data[cols].to_numpy(dtype=float)
P_lev, Z = collapse(P_train)
L = len(P_lev)

# --- profile the single shared rate -----------------------------------------
def neg_LL(r):
    if r <= 0:
        return 1e10
    fit = _mixed_solve(Z, kernel(P_lev, [r]), y, X)
    return -float(fit.rx2("LL")[0])

res = minimize_scalar(neg_LL, bounds=(1e-6, 10.0), method="bounded")
rho = [float(res.x)]

K = kernel(P_lev, rho)
fit = _mixed_solve(Z, K, y, X, se=True)
Vu = float(fit.rx2("Vu")[0])
Ve = float(fit.rx2("Ve")[0])
beta = float(np.asarray(fit.rx2("beta")).ravel()[0])
beta_se = float(np.asarray(fit.rx2("beta.SE")).ravel()[0])
blup = np.asarray(fit.rx2("u")).ravel().tolist()
pev_diag = (np.asarray(fit.rx2("u.SE")).ravel() ** 2).tolist()

with open("../data/spat_iso.json", "w") as f:
    json.dump({
        "rho": rho, "Vu": Vu, "Ve": Ve, "beta": beta,
        "eev_intercept": beta_se ** 2, "blup": blup, "pev_diag": pev_diag,
    }, f, indent=2)

P_pred = holdout[cols].to_numpy(dtype=float) + PRED_OFFSET
m = len(P_pred)
K_full = kernel(np.vstack([P_lev, P_pred]), rho)
Z_aug = np.hstack([Z, np.zeros((n, m))])
fit2 = _mixed_solve(Z_aug, K_full, y, X)
blup_pred = np.asarray(fit2.rx2("u")).ravel()[L:].tolist()

with open("../data/spat_iso_pred.json", "w") as f:
    json.dump({"n_train": L, "n_pred": m, "blup_pred": blup_pred}, f, indent=2)

print(f"spat_iso: rho={rho} Vu={Vu:.4g} Ve={Ve:.4g} L={L} m={m}")
prediction_map("iso", data, y, X, n, cols, kernel, rho, P_lev, Z, L)


# %% [markdown]
# ## `spat_ani` — separable AR, anisotropic
# One rate per axis: elliptical (axis-aligned) contours.

# %%
data = _grid(DF)
holdout = _holdout(data)
cols = ["X", "Y"]
kernel = separable_ani_K

y = data["height"].to_numpy()
n = len(y)
X = np.ones((n, 1))

P_train = data[cols].to_numpy(dtype=float)
P_lev, Z = collapse(P_train)
L = len(P_lev)

# --- profile one rate per axis (Nelder-Mead) --------------------------------
def neg_LL(rho_vec):
    rho_vec = np.atleast_1d(rho_vec).astype(float)
    if np.any(rho_vec <= 0):
        return 1e10
    fit = _mixed_solve(Z, kernel(P_lev, rho_vec), y, X)
    return -float(fit.rx2("LL")[0])

res = minimize(neg_LL, x0=np.ones(2), method="Nelder-Mead")
rho = [float(v) for v in res.x]

K = kernel(P_lev, rho)
fit = _mixed_solve(Z, K, y, X, se=True)
Vu = float(fit.rx2("Vu")[0])
Ve = float(fit.rx2("Ve")[0])
beta = float(np.asarray(fit.rx2("beta")).ravel()[0])
beta_se = float(np.asarray(fit.rx2("beta.SE")).ravel()[0])
blup = np.asarray(fit.rx2("u")).ravel().tolist()
pev_diag = (np.asarray(fit.rx2("u.SE")).ravel() ** 2).tolist()

with open("../data/spat_ani.json", "w") as f:
    json.dump({
        "rho": rho, "Vu": Vu, "Ve": Ve, "beta": beta,
        "eev_intercept": beta_se ** 2, "blup": blup, "pev_diag": pev_diag,
    }, f, indent=2)

P_pred = holdout[cols].to_numpy(dtype=float) + PRED_OFFSET
m = len(P_pred)
K_full = kernel(np.vstack([P_lev, P_pred]), rho)
Z_aug = np.hstack([Z, np.zeros((n, m))])
fit2 = _mixed_solve(Z_aug, K_full, y, X)
blup_pred = np.asarray(fit2.rx2("u")).ravel()[L:].tolist()

with open("../data/spat_ani_pred.json", "w") as f:
    json.dump({"n_train": L, "n_pred": m, "blup_pred": blup_pred}, f, indent=2)

print(f"spat_ani: rho={rho} Vu={Vu:.4g} Ve={Ve:.4g} L={L} m={m}")
prediction_map("ani", data, y, X, n, cols, kernel, rho, P_lev, Z, L)