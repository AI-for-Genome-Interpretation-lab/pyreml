# %%
import pandas as pd
import numpy as np
import json
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar

from pyreml import MixedModel, Random, larix as df

rrBLUP = importr("rrBLUP")

df = df[df["year"] == 2000]
df["ID"] = np.arange(len(df))

# %%
coords_train = df[["X", "Y"]].values
diff = coords_train[:, np.newaxis, :] - coords_train[np.newaxis, :, :]
D_train = np.sqrt((diff ** 2).sum(axis=-1))

plt.figure(figsize=(4,3.5))
sns.heatmap(
    D_train,
    cmap='coolwarm',
    center=0,
    annot=False,
)
plt.tight_layout()
plt.show()

y = df["height"].to_numpy()
n = len(y)
Zmat = np.eye(n)
Xmat = np.ones((n, 1))

# %%
def neg_LL(rho):
    K = np.exp(-rho * D_train)
    with localconverter(ro.default_converter + numpy2ri.converter):
        K_r = ro.conversion.py2rpy(K)
        y_r = ro.conversion.py2rpy(y)
        Z_r = ro.conversion.py2rpy(Zmat)
        X_r = ro.conversion.py2rpy(Xmat)
    fit = rrBLUP.mixed_solve(y=y_r, Z=Z_r, K=K_r, X=X_r, method="REML")
    LL = float(fit.rx2("LL")[0])
    return -LL

res = minimize_scalar(neg_LL, bounds=(1e-6, 10.0), method="bounded")

rho = res.x
rho

# %%
mod_ar = MixedModel.from_dataframe(
        data=df,
        response="height",
        fixed="1",
        random=Random(
            unit="ID",
            right_hand="ar",
            distance=D_train,
            matrix_index=df["ID"].tolist(),
        ),
    ).fit()
print(mod_ar.random[0].variance["metadata"]["ar"])
print(mod_ar.random[0].variance["sigma"])
print(mod_ar.residual.variance["sigma"])

# %%
mod_ar = MixedModel.from_dataframe(
        data=df,
        response="height",
        fixed="1",
        random=Random(
            unit="ID",
            right_hand="ar",
            distance=D_train,
            matrix_index=df["ID"].tolist(),
        ),
    )

print(mod_ar.beta.item())
mod_ar.OLS()
print(mod_ar.beta.item())
mod_ar.REML()
print(mod_ar.beta.item())

# %%
with localconverter(ro.default_converter + numpy2ri.converter):
    K_r = ro.conversion.py2rpy(np.exp(-rho * D_train))
    y_r = ro.conversion.py2rpy(y)
    Z_r = ro.conversion.py2rpy(Zmat)
    X_r = ro.conversion.py2rpy(Xmat)

fit = rrBLUP.mixed_solve(y=y_r, Z=Z_r, K=K_r, X=X_r, method="REML",SE=True)

Vu = float(fit.rx2("Vu")[0])
Ve = float(fit.rx2("Ve")[0])
beta = float(np.asarray(fit.rx2("beta")).ravel()[0])
beta_se = float(np.asarray(fit.rx2("beta.SE")).ravel()[0])
eev_intercept = beta_se ** 2
blup = np.asarray(fit.rx2("u")).ravel().tolist()          # ID order 0..n-1
pev_diag = (np.asarray(fit.rx2("u.SE")).ravel() ** 2).tolist()

print(Vu)
print(Ve)

spatial = {
    "rho": rho,
    "Vu": Vu,
    "Ve": Ve,
    "beta": beta,
    "eev_intercept": eev_intercept,
    "blup": blup,
    "pev_diag": pev_diag,
}
 
with open("../data/spatial.json", "w") as f:
    json.dump(spatial, f, indent=2)


# %% [markdown]
# ## Prediction

# %%
pred = df[
    (df["BLOC"].isin(["B13", "B14", "B15", "B16"])) &
    (df["X"].notna()) &
    (df["Y"].notna())
].copy()
 
m = len(pred)
print("n_train:", n, "n_pred:", m)
 
coords_pred = pred[["X", "Y"]].to_numpy()
coords_full = np.vstack([coords_train, coords_pred])
diff = coords_full[:, np.newaxis, :] - coords_full[np.newaxis, :, :]
D_full = np.sqrt((diff ** 2).sum(axis=-1))

K_full = np.exp(-rho * D_full)
Z_aug = np.hstack([np.eye(n), np.zeros((n, m))])   # n x (n+m)
X_aug = np.ones((n, 1))
 
with localconverter(ro.default_converter + numpy2ri.converter):
    K_r = ro.conversion.py2rpy(K_full)
    y_r = ro.conversion.py2rpy(y)
    Z_r = ro.conversion.py2rpy(Z_aug)
    X_r = ro.conversion.py2rpy(X_aug)
 
fit2 = rrBLUP.mixed_solve(y=y_r, Z=Z_r, K=K_r, X=X_r, method="REML")
 
u_full = np.asarray(fit2.rx2("u")).ravel()
blup_pred = u_full[n:].tolist()              # pred levels n..n+m-1, = pred row order
print("len(blup_pred):", len(blup_pred))

spatial_pred = {
    "n_train": n,
    "n_pred": m,
    "blup_pred": blup_pred,
}
 
with open("../data/spatial_pred.json", "w") as f:
    json.dump(spatial_pred, f, indent=2)



