# %%
import pandas as pd
import numpy as np
import torch

import json
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from pyreml import MixedModel, Random, Residual, larix as df, prepare_pedigree

nadiv = importr("nadiv")
stats = importr("stats")
breedR = importr("breedR")

def nearPD(S, eps: float = 1e-8):
    """
    Nearest positive-definite surrogate of a symmetric matrix.

    Symmetrize, clamp the eigenvalues at `eps`, rebuild, symmetrize again.
    Guarantees strictly positive eigenvalues, which is what the factor-analytic
    initialization relies on (the q dominant eigenvalues feed Lambda > 0).
    """
    was_numpy = isinstance(S, np.ndarray)
    S = torch.as_tensor(S, dtype=torch.double)
    S = (S + S.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clamp(eigvals, min=eps)
    S = (eigvecs * eigvals) @ eigvecs.T
    S = (S + S.T) / 2
    return S.numpy() if was_numpy else S

df = df[df["year"] == 2000]

df_train = df[
    df["BLOC"].isin(["B1","B2","B3","B4","B5","B6","B7","B8"])
]

df_tot = df[
    df["BLOC"].isin(["B1","B2","B3","B4","B5","B6","B7","B8",
                     "B9","B10","B11","B12"])
]

pedigree = prepare_pedigree(df_tot[["ID","DAM","SIRE"]])

pedigree

# %%
with localconverter(ro.default_converter + pandas2ri.converter):
    ped_r = ro.conversion.py2rpy(pedigree[["id", "dam", "sire"]])

A_sparse = nadiv.makeA(ped_r)
D_sparse = nadiv.makeD(ped_r)

A = np.array(ro.r("as.matrix")(A_sparse))
D_result = nadiv.makeD(ped_r)
D_sparse = D_result.rx2("D")
D = np.array(ro.r("as.matrix")(D_sparse))

print("A shape:", A.shape, "D shape:", D.shape)
print("A diag range:", A.diagonal().min(), "-", A.diagonal().max())
print("D diag range:", D.diagonal().min(), "-", D.diagonal().max())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(A, cmap="coolwarm", center=0, ax=axes[0], square=True)
axes[0].set_title("A (additive)")
sns.heatmap(D, cmap="coolwarm", center=0, ax=axes[1], square=True)
axes[1].set_title("D (dominance)")
plt.tight_layout()
plt.show()

# Keep only founders + observed individuals (B1-B12)
ped_ids = pedigree["id"].tolist()
keep_set = set(df_train["ID"].unique())
keep_idx = [i for i, id_ in enumerate(ped_ids) if id_ in keep_set]

A_keep = A[np.ix_(keep_idx, keep_idx)]
D_keep = D[np.ix_(keep_idx, keep_idx)]
ped_ids_keep = [ped_ids[i] for i in keep_idx]

print(f"full pedigree: {len(ped_ids)}, kept: {len(ped_ids_keep)}")

kinship = {
    "index": ped_ids_keep,
    "A": A_keep.tolist(),
    "D": D_keep.tolist(),
}

with open("../data/pedigree_kinship.json", "w") as f:
    json.dump(kinship, f, indent=2)

# %%
id_to_idx = {id_: i for i, id_ in enumerate(ped_ids)}

Z_flo = np.zeros((len(df_train), len(ped_ids)))
for i, obs_id in enumerate(df_train["ID"].values):
    Z_flo[i, id_to_idx[obs_id]] = 1

print("Z shape:", Z_flo.shape, "  (obs x pedigree levels)")
print("non-zero cols:", int(Z_flo.sum(axis=0).astype(bool).sum()), "/", len(ped_ids))

# %%
mod = MixedModel.from_dataframe(
    data=df_train,
    response="flexuosity",
    fixed="1 + BLOC",
    random=[
        Random(unit="ID", right_hand="str", covariance=A, matrix_index=ped_ids),
        Random(unit="ID", right_hand="str", covariance=D, matrix_index=ped_ids),
    ],
).fit()

# %%
with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    Z_r    = ro.conversion.py2rpy(Z_flo)
    Ainv_r = ro.conversion.py2rpy(np.linalg.inv(A))
    Dinv_r = ro.conversion.py2rpy(np.linalg.inv(D))
    tab_r  = ro.conversion.py2rpy(df_train)

# %%
mod_uni_em = breedR.remlf90(
    stats.formula("flexuosity ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   ro.r("matrix")(1.0, 1, 1),
        }),
        "D": ro.ListVector({
            "incidence": Z_r,
            "precision": Dinv_r,
            "var.ini":   ro.r("matrix")(1.0, 1, 1),
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": ro.r("matrix")(1.0, 1, 1),
    })},
    method="em",
    data=tab_r,
)

with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    # --- Variances ---
    var_a = float(np.array(mod_uni_em.rx2("var")[0]).ravel()[0])
    var_d = float(np.array(mod_uni_em.rx2("var")[1]).ravel()[0])
    var_r = float(np.array(mod_uni_em.rx2("var")[2]).ravel()[0])
    print("Var_A:", var_a, "  Var_D:", var_d, "  Var_R:", var_r)

print(f"var_a: pyreml={float(torch.exp(mod.random[0].log_S))}")
print(f"var_d: pyreml={float(torch.exp(mod.random[1].log_S))}")
print(f"var_r: pyreml={float(torch.exp(mod.residual.log_S))}")

# %%
mod_uni_ai = breedR.remlf90(
    stats.formula("flexuosity ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   ro.r("matrix")(1.0, 1, 1),
        }),
        "D": ro.ListVector({
            "incidence": Z_r,
            "precision": Dinv_r,
            "var.ini":   ro.r("matrix")(1.0, 1, 1),
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": ro.r("matrix")(1.0, 1, 1),
    })},
    method="ai",
    data=tab_r,
)

with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    # --- Variances ---
    var_a = float(np.array(mod_uni_ai.rx2("var")[0]).ravel()[0])
    var_d = float(np.array(mod_uni_ai.rx2("var")[1]).ravel()[0])
    var_r = float(np.array(mod_uni_ai.rx2("var")[2]).ravel()[0])
    print("Var_A:", var_a, "  Var_D:", var_d, "  Var_R:", var_r)

print(f"var_a: pyreml={float(torch.exp(mod.random[0].log_S))}")
print(f"var_d: pyreml={float(torch.exp(mod.random[1].log_S))}")
print(f"var_r: pyreml={float(torch.exp(mod.residual.log_S))}")

# %%
mod_uni_ai_init = breedR.remlf90(
    stats.formula("flexuosity ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   ro.r("matrix")(float(torch.exp(mod.random[0].log_S)), 1, 1),
        }),
        "D": ro.ListVector({
            "incidence": Z_r,
            "precision": Dinv_r,
            "var.ini":   ro.r("matrix")(float(torch.exp(mod.random[1].log_S)), 1, 1),
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": ro.r("matrix")(float(torch.exp(mod.residual.log_S)), 1, 1),
    })},
    method="ai",
    data=tab_r,
)

with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    # --- Variances ---
    var_a = float(np.array(mod_uni_ai.rx2("var")[0]).ravel()[0])
    var_d = float(np.array(mod_uni_ai.rx2("var")[1]).ravel()[0])
    var_r = float(np.array(mod_uni_ai.rx2("var")[2]).ravel()[0])
    print("Var_A:", var_a, "  Var_D:", var_d, "  Var_R:", var_r)

print(f"var_a: pyreml={float(torch.exp(mod.random[0].log_S))}")
print(f"var_d: pyreml={float(torch.exp(mod.random[1].log_S))}")
print(f"var_r: pyreml={float(torch.exp(mod.residual.log_S))}")

# %%
print(mod_uni_em.rx2("fit").rx2("-2logL"))
print(mod_uni_ai.rx2("fit").rx2("-2logL"))
print(mod_uni_ai_init.rx2("fit").rx2("-2logL"))
assert mod_uni_ai_init.rx2("fit").rx2("-2logL")[0] < mod_uni_ai.rx2("fit").rx2("-2logL")[0]
assert mod_uni_ai_init.rx2("fit").rx2("-2logL")[0] < mod_uni_em.rx2("fit").rx2("-2logL")[0]
mod_uni = mod_uni_ai_init

# %%
conv_uni = float(mod_uni.rx2("reml").rx2("convergence")[0]) < 1e-8
print("Converged:", conv_uni)

with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    # --- Variances ---
    var_a = float(np.array(mod_uni.rx2("var")[0]).ravel()[0])
    var_d = float(np.array(mod_uni.rx2("var")[1]).ravel()[0])
    var_r = float(np.array(mod_uni.rx2("var")[2]).ravel()[0])
    print("Var_A:", var_a, "  Var_D:", var_d, "  Var_R:", var_r)

    # --- Fixed effects (named vectors with attr "se") ---
    fixef_raw = ro.r("fixef")(mod_uni)

    intercept = float(np.array(ro.r("`[[`")(fixef_raw, "Intercept"))[0])
    bloc_vals = np.array(ro.r("`[[`")(fixef_raw, "BLOC")).ravel().tolist()

    beta_uni = [intercept] + bloc_vals

    print("Fix effects:", beta_uni)

    # --- Residuals ---
    resid_uni = np.array(ro.r("residuals")(mod_uni)).ravel().tolist()

# R helpers — extraction complète côté R, un seul vecteur revient en Python
extract_blup = ro.r('function(mod, effect) mod$ranef[[effect]][[1]][["value"]]')
extract_se   = ro.r('function(mod, effect) mod$ranef[[effect]][[1]][["s.e."]]')

# --- Identify train vs pred-only varieties ---
train_ids_set = set(df_train["ID"].unique())
pred_ids_set  = set(df_tot["ID"].unique()) - train_ids_set

train_variety_ids = [id_ for id_ in ped_ids if id_ in train_ids_set]
pred_variety_ids  = [id_ for id_ in ped_ids if id_ in pred_ids_set]

train_idx = [ped_ids.index(id_) for id_ in train_variety_ids]
pred_idx  = [ped_ids.index(id_) for id_ in pred_variety_ids]

print(f"train varieties: {len(train_variety_ids)}, "
      f"pred varieties: {len(pred_variety_ids)}, "
      f"founders: {len(ped_ids) - len(train_variety_ids) - len(pred_variety_ids)}")

# --- Full extraction ---
blup_a = list(extract_blup(mod_uni, "A"))
se_a   = list(extract_se(mod_uni, "A"))
blup_d = list(extract_blup(mod_uni, "D"))
se_d   = list(extract_se(mod_uni, "D"))

# --- Save with train/pred split ---
pedigree_uni = {
    "var_a": var_a,
    "var_d": var_d,
    "var_r": var_r,
    "beta": beta_uni,
    "ped_index": ped_ids,
    # full BLUPs (all pedigree, ped_index order)
    "blup_a": blup_a,
    "se_a": se_a,
    "blup_d": blup_d,
    "se_d": se_d,
    "residuals": resid_uni,
    # prediction subset (IDs in B13-B16, absent from B1-B12)
    "train_ids": train_variety_ids,
    "pred_ids": pred_variety_ids,
    "blup_a_pred": [blup_a[i] for i in pred_idx],
    "blup_d_pred": [blup_d[i] for i in pred_idx],
    "se_a_pred": [se_a[i] for i in pred_idx],
    "se_d_pred": [se_d[i] for i in pred_idx],
}

with open("../data/pedigree_uni.json", "w") as f:
    json.dump(pedigree_uni, f, indent=2)

# %%
traits = ["height", "circumference", "flexuosity"]

uni_var_a = {}
uni_var_r = {}

for trait in traits:
    print(f"\n--- {trait} ---")
    mod = breedR.remlf90(
        stats.formula(f"{trait} ~ 1 + BLOC"),
        generic=ro.ListVector({
            "A": ro.ListVector({
                "incidence": Z_r,
                "precision": Ainv_r,
                "var.ini":   ro.r("matrix")(1.0, 1, 1),
            }),
        }),
        **{"var.ini": ro.ListVector({
            "resid": ro.r("matrix")(1.0, 1, 1),
        })},
        method="ai",
        data=tab_r,
    )

    conv = float(mod.rx2("reml").rx2("convergence")[0]) < 1e-8
    with localconverter(ro.default_converter + numpy2ri.converter):
        va = float(np.array(mod.rx2("var")[0]).ravel()[0])
        vr = float(np.array(mod.rx2("var")[1]).ravel()[0])

    uni_var_a[trait] = va
    uni_var_r[trait] = vr
    print(f"  converged: {conv}  Var_A: {va:.4f}  Var_R: {vr:.4f}")

print("\nSummary:")
for t in traits:
    print(f"  {t:15s}  Var_A={uni_var_a[t]:.4f}  Var_R={uni_var_r[t]:.4f}")

# %%
SigmaA_diag = np.diag([uni_var_a[t] for t in traits])
SigmaR_diag = np.diag([uni_var_r[t] for t in traits])

print("SigmaA_diag init:\n", SigmaA_diag)
print("SigmaR_diag init:\n", SigmaR_diag)

with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    SigmaA_r = ro.conversion.py2rpy(SigmaA_diag)
    SigmaR_r = ro.conversion.py2rpy(SigmaR_diag)

mod_diag = breedR.remlf90(
    stats.formula(f"cbind({', '.join(traits)}) ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   SigmaA_r,
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": SigmaR_r,
    })},
    method="ai",
    data=tab_r,
)

conv_diag = float(mod_diag.rx2("reml").rx2("convergence")[0]) < 1e-8
print("Converged:", conv_diag)

with localconverter(ro.default_converter + numpy2ri.converter):
    SigmaA_hat = np.array(mod_diag.rx2("var")[0])
    SigmaR_hat = np.array(mod_diag.rx2("var")[1])

print("SigmaA_hat:\n", SigmaA_hat)
print("SigmaR_hat:\n", SigmaR_hat)


# %%
# --- R helpers (multivariate) ---
extract_ranef_multi = ro.r('function(mod, eff) mod$ranef[[eff]]')
extract_intercept   = ro.r('function(mod) fixef(mod)[["Intercept"]]')
extract_bloc        = ro.r('function(mod) fixef(mod)[["BLOC"]]')

# --- Extraction ---
with localconverter(ro.default_converter + numpy2ri.converter):
    ranef_raw     = np.array(extract_ranef_multi(mod_diag, "A"))   # (n_ped, 2, n_traits)
    intercept_vec = np.array(extract_intercept(mod_diag)).ravel()  # (n_traits,)
    bloc_mat      = np.array(extract_bloc(mod_diag))               # (12, 3) — check .T if needed
    resid_diag    = np.array(ro.r("residuals")(mod_diag)).ravel()

# ranef_raw: shape (3, 190), each element is a (value, se) tuple
blup_a_mat = np.array([[x[0] for x in col] for col in ranef_raw]).T   # (190, 3)
se_a_mat   = np.array([[x[1] for x in col] for col in ranef_raw]).T   # (190, 3)

blup_ids   = ped_ids
bloc_names = sorted(df_train["BLOC"].unique())

# --- Train/pred split ---
blup_id_to_idx = {id_: i for i, id_ in enumerate(blup_ids)}
train_blup_idx = [blup_id_to_idx[id_] for id_ in train_variety_ids if id_ in blup_id_to_idx]
pred_blup_idx  = [blup_id_to_idx[id_] for id_ in pred_variety_ids  if id_ in blup_id_to_idx]

print(f"ranef shape: {ranef_raw.shape}")
print(f"blup_a: {blup_a_mat.shape}, se_a: {se_a_mat.shape}")
print(f"intercept: {intercept_vec}")
print(f"bloc shape: {bloc_mat.shape}")
print(f"train: {len(train_blup_idx)}, pred: {len(pred_blup_idx)}")
print(f"residuals: {len(resid_diag)}")

# --- Save ---
pedigree_diag = {
    "traits": traits,
    "SigmaA": SigmaA_hat.tolist(),
    "SigmaR": SigmaR_hat.tolist(),
    "beta_intercept": intercept_vec.tolist(),
    "beta_bloc": bloc_mat.tolist(),
    "bloc_names": bloc_names,
    "blup_index": blup_ids,
    "blup_a": blup_a_mat.tolist(),
    "se_a_train": se_a_mat[train_blup_idx].tolist(),   # PEV train only
    "residuals": resid_diag.tolist(),
    "train_ids": train_variety_ids,
    "pred_ids": pred_variety_ids,
    "blup_a_pred": blup_a_mat[pred_blup_idx].tolist(),
}

with open("../data/pedigree_diag.json", "w") as f:
    json.dump(pedigree_diag, f, indent=2)

# %%
pairs = list(combinations(traits, 2))
biv_cov_a = {}
biv_cov_r = {}

for t1, t2 in pairs:
    print(f"\n--- {t1} x {t2} ---")

    # Initial covariance matrices (2x2)
    va1, va2 = uni_var_a[t1], uni_var_a[t2]
    vr1, vr2 = uni_var_r[t1], uni_var_r[t2]

    cov_a_init = np.array([
        [va1, 0.5 * np.sqrt(va1 * va2)],
        [0.5 * np.sqrt(va1 * va2), va2],
    ])
    cov_r_init = np.array([
        [vr1, 0.5 * np.sqrt(vr1 * vr2)],
        [0.5 * np.sqrt(vr1 * vr2), vr2],
    ])

    with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
        Sa_r = ro.conversion.py2rpy(cov_a_init)
        Sr_r = ro.conversion.py2rpy(cov_r_init)

    mod_biv = breedR.remlf90(
        stats.formula(f"cbind({t1}, {t2}) ~ 1 + BLOC"),
        generic=ro.ListVector({
            "A": ro.ListVector({
                "incidence": Z_r,
                "precision": Ainv_r,
                "var.ini":   Sa_r,
            }),
        }),
        **{"var.ini": ro.ListVector({
            "resid": Sr_r,
        })},
        method="em",
        data=tab_r,
    )

    conv = float(mod_biv.rx2("reml").rx2("convergence")[0]) < 1e-8
    with localconverter(ro.default_converter + numpy2ri.converter):
        Sa_hat = np.array(mod_biv.rx2("var")[0])
        Sr_hat = np.array(mod_biv.rx2("var")[1])

    biv_cov_a[(t1, t2)] = Sa_hat
    biv_cov_r[(t1, t2)] = Sr_hat

    print(f"  converged: {conv}")
    print(f"  SigmaA:\n{Sa_hat}")
    print(f"  SigmaR:\n{Sr_hat}")

# %%
def assemble(biv_cov, uni_var, traits):
    """Build a covariance matrix from bivariate 2x2 estimates."""
    n = len(traits)
    S = np.zeros((n, n))
    idx = {t: i for i, t in enumerate(traits)}

    # Diagonal from bivariates (average of the two estimates per trait)
    diag_accum = {t: [] for t in traits}
    for (t1, t2), cov in biv_cov.items():
        diag_accum[t1].append(cov[0, 0])
        diag_accum[t2].append(cov[1, 1])
    for t in traits:
        S[idx[t], idx[t]] = np.mean(diag_accum[t])

    # Off-diagonal from bivariates
    for (t1, t2), cov in biv_cov.items():
        i, j = idx[t1], idx[t2]
        S[i, j] = cov[0, 1]
        S[j, i] = cov[1, 0]

    return S

SigmaA_init = assemble(biv_cov_a, uni_var_a, traits)
SigmaR_init = assemble(biv_cov_r, uni_var_r, traits)

SigmaA_init = nearPD(SigmaA_init)
SigmaR_init = nearPD(SigmaR_init)

print("SigmaA_init:\n", SigmaA_init)
print("SigmaR_init:\n", SigmaR_init)

# Check PD (required for remlf90)
print("SigmaA eigvals:", np.linalg.eigvalsh(SigmaA_init))
print("SigmaR eigvals:", np.linalg.eigvalsh(SigmaR_init))


# %%
with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    SigmaA_r = ro.conversion.py2rpy(SigmaA_init)
    SigmaR_r = ro.conversion.py2rpy(SigmaR_init)

# %%
mod_str_ai = breedR.remlf90(
    stats.formula(f"cbind({', '.join(traits)}) ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   SigmaA_r,
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": SigmaR_r,
    })},
    method="ai",
    data=tab_r,
)

conv_str_ai = float(mod_str_ai.rx2("reml").rx2("convergence")[0]) < 1e-8
print("Converged:", conv_str_ai)

with localconverter(ro.default_converter + numpy2ri.converter):
    SigmaA_str_ai = np.array(mod_str_ai.rx2("var")[0])
    SigmaR_str_ai = np.array(mod_str_ai.rx2("var")[1])

# %%
mod_str_em = breedR.remlf90(
    stats.formula(f"cbind({', '.join(traits)}) ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   SigmaA_r,
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": SigmaR_r,
    })},
    method="em",
    data=tab_r,
)

conv_str_em = float(mod_str_em.rx2("reml").rx2("convergence")[0]) < 1e-8
print("Converged:", conv_str_em)

with localconverter(ro.default_converter + numpy2ri.converter):
    SigmaA_str_em = np.array(mod_str_em.rx2("var")[0])
    SigmaR_str_em = np.array(mod_str_em.rx2("var")[1])

# %%
mod_pyreml = MixedModel.from_dataframe(
    data=df_train,
    response= traits,
    fixed="1 + BLOC",
    random=Random(
        unit="ID",
        left_hand="full",
        right_hand="str",
        covariance=A,
        matrix_index=ped_ids,
        init=SigmaA_init,
    ),
    residual=Residual(
        left_hand="full",
        init=SigmaR_init,
    ),
).fit()

# %%
print("=== SigmaA ===")
print("pyreml:\n", mod_pyreml.random[0].build_S().detach().numpy())
print("breedR_ai:\n", SigmaA_str_ai)
print("breedR_em:\n", SigmaA_str_em)

print("\n=== SigmaR ===")
print("pyreml:\n", mod_pyreml.residual.build_S().detach().numpy())
print("breedR_ai:\n", SigmaR_str_ai)
print("breedR_em:\n", SigmaR_str_em)

# %%
Sa = mod_pyreml.random[0].build_S().detach().numpy()
Sr = mod_pyreml.residual.build_S().detach().numpy()
with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    SigmaA_r = ro.conversion.py2rpy((Sa + Sa.T)/2 + np.eye(len(traits)) * 1e-8)
    SigmaR_r = ro.conversion.py2rpy((Sr + Sr.T)/2 + np.eye(len(traits)) * 1e-8)

mod_str_em_init = breedR.remlf90(
    stats.formula(f"cbind({', '.join(traits)}) ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   SigmaA_r,
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": SigmaR_r,
    })},
    method="em",
    data=tab_r,
)

mod_str_ai_init = breedR.remlf90(
    stats.formula(f"cbind({', '.join(traits)}) ~ 1 + BLOC"),
    generic=ro.ListVector({
        "A": ro.ListVector({
            "incidence": Z_r,
            "precision": Ainv_r,
            "var.ini":   SigmaA_r,
        }),
    }),
    **{"var.ini": ro.ListVector({
        "resid": SigmaR_r,
    })},
    method="ai",
    data=tab_r,
)

# %%
assert np.isnan(mod_str_ai.rx2("fit").rx2("-2logL")[0])
assert np.isnan(mod_str_ai_init.rx2("fit").rx2("-2logL")[0])
assert mod_str_em_init.rx2("fit").rx2("-2logL")[0] < mod_str_em.rx2("fit").rx2("-2logL")[0]
print(mod_str_em.rx2("fit").rx2("-2logL"))
print(mod_str_em_init.rx2("fit").rx2("-2logL"))
mod_str = mod_str_em_init

# %%
# --- breedR extraction (kept as cross-check, no longer saved) ---
with localconverter(ro.default_converter + numpy2ri.converter):
    SigmaA_str = np.array(mod_str.rx2("var")[0])
    SigmaR_str = np.array(mod_str.rx2("var")[1])
    ranef_raw_str     = np.array(extract_ranef_multi(mod_str, "A"))
    intercept_vec_str = np.array(extract_intercept(mod_str)).ravel()
    bloc_mat_str      = np.array(extract_bloc(mod_str))
    resid_str         = np.array(ro.r("residuals")(mod_str)).ravel()

print("SigmaA_str:\n", SigmaA_str)
print("SigmaR_str:\n", SigmaR_str)

blup_a_str = np.array([[x[0] for x in col] for col in ranef_raw_str]).T   # (190, 3)
se_a_str   = np.array([[x[1] for x in col] for col in ranef_raw_str]).T   # (190, 3)

# --- Save ---
pedigree_str = {
    "traits": traits,
    "SigmaA": SigmaA_str.tolist(),
    "SigmaR": SigmaR_str.tolist(),
    "beta_intercept": intercept_vec_str.ravel().tolist(),
    "beta_bloc": bloc_mat_str.tolist(),
    "bloc_names": bloc_names,
    "blup_index": ped_ids,
    "blup_a": blup_a_str.tolist(),
    "se_a_train": se_a_str[train_blup_idx].tolist(),
    "residuals": resid_str.tolist(),
    "train_ids": train_variety_ids,
    "pred_ids": pred_variety_ids,
    "blup_a_pred": blup_a_str[pred_blup_idx].tolist(),
}

with open("../data/pedigree_str.json", "w") as f:
    json.dump(pedigree_str, f, indent=2)


