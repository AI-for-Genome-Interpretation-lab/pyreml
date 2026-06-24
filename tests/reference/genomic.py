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
from scipy.stats import norm
from scipy.special import softmax

from pyreml import MixedModel, Random, Residual, A_pedigree, larix as df

rrBLUP = importr("rrBLUP")

for col in ["SIRE", "DAM"]:
    df[col] = df[col].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)

rng = np.random.default_rng(42) 

# %%
df = df[
    df["BLOC"].isin(["B1","B2","B3","B4","B5","B6","B7","B8"]) &
    (df["year"] == 2000)
]

pedigree = df[["ID","DAM","SIRE"]]
print(len(pedigree))
pedigree = pedigree.drop_duplicates(subset="ID")
print(len(pedigree))

all_parents = set(pedigree["DAM"]).union(pedigree["SIRE"]) - {np.nan}
founders = all_parents - set(pedigree["ID"])
founder_rows = pd.DataFrame({"ID": list(founders), "DAM": np.nan, "SIRE": np.nan})
pedigree = pd.concat([founder_rows, pedigree], ignore_index=True)
print(len(pedigree))

A = A_pedigree(pedigree)

# %% [markdown]
# ## Simulate a genomic matrix from pedigree

# %%
n = A.shape[0]
m = 500
maf = rng.uniform(0.05, 0.95, size=m)
thr = norm.ppf(1 - maf)

L = np.linalg.cholesky(A)

X = np.zeros((n, m))
for l in range(m):
    g1 = (L @ rng.standard_normal(n)) > thr[l]
    g2 = (L @ rng.standard_normal(n)) > thr[l]
    X[:, l] = (g1.astype(float) - 0.5) + (g2.astype(float) - 0.5)  #

with localconverter(ro.default_converter + numpy2ri.converter):
    Xr = ro.conversion.py2rpy(X)

K2r = rrBLUP.A_mat(
    X = Xr,
    min_MAF = 0.05,
    max_missing = 1,
    shrink = True,
)
K2 = numpy2ri.rpy2py(K2r)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(A, cmap="coolwarm", center=0, ax=axes[0], square=True)
axes[0].set_title("True pedigree kinship")
sns.heatmap(K2, cmap="coolwarm", center=0, ax=axes[1], square=True)
axes[1].set_title("Simulated genomic kinship")
plt.tight_layout()
plt.show()

K = K2

# %% [markdown]
# ## Simulate a multritrial scenario

# %%
p = 5
Sigma_A_scale = 15

m = rng.normal(100, 20, size = p)
print(m)

Sigma_R = rng.gamma(scale = 5, shape = 1, size = p)
print(Sigma_R)

Lambda = softmax(np.arange(p))[::-1]
Lambda = Lambda / sum(Lambda) * Sigma_A_scale * p
M = rng.standard_normal((p, p))
Q, _ = np.linalg.qr(M)
Sigma_A = Q @ np.diag(Lambda) @ Q.T

sns.heatmap(Sigma_A, cmap="coolwarm", center = 0, square=True)
plt.tight_layout()
plt.show()

Ls = np.linalg.cholesky(Sigma_A).T
Lg = np.linalg.cholesky(K).T
    
z = rng.normal(0,1, size = p * n)
z = np.reshape(z, (p, n))
u = Ls @ z @ Lg.T
u = u.T
print(u.shape)

# %%
id_to_row = {id_: i for i, id_ in enumerate(pedigree["ID"])}
rows = df["ID"].map(id_to_row).to_numpy()

sim = df[["ID"]].copy()
for envt in range(p):
    sim[f"u_{envt}"] = u[rows, envt]

for envt in range(p):
    resid = rng.normal(0, np.sqrt(Sigma_R[envt]), size=len(sim))
    sim[f"y_{envt}"] = m[envt] + sim[f"u_{envt}"] + resid

print(sim)

df_tmp = sim.drop(columns=[f"u_{envt}" for envt in range(p)]).melt(
    id_vars="ID",
    value_vars=[f"y_{envt}" for envt in range(p)],
    var_name="envt",
    value_name="y",
)
df_tmp["envt"] = df_tmp["envt"].str.removeprefix("y_").astype(int)

df_cols = df_tmp.copy()
for envt in range(p):
    df_cols[f"y_{envt}"] = np.where(df_cols["envt"] == envt, df_cols["y"], np.nan)
df_cols = df_cols.drop(columns=["y", "envt"])
df_cols

# %%
sim_simplified = df[["ID"]].copy()
sim_simplified["u"] = u[rows, 0]

for envt in range(p):
    resid = rng.normal(0, np.sqrt(Sigma_R[envt]), size=len(sim_simplified))
    sim_simplified[f"y_{envt}"] = m[envt] + sim_simplified["u"] + resid

print(sim_simplified)

df_long = sim_simplified.drop(columns=["u"]).melt(
    id_vars="ID",
    value_vars=[f"y_{envt}" for envt in range(p)],
    var_name="envt",
    value_name="y",
)
df_long["envt"] = df_long["envt"].str.removeprefix("y_").astype(int)
df_long["envt"] = df_long["envt"].astype("category")

print(df_long)

# %% [markdown]
# ## Save all artifacts for the test suite

# %%
def _df_records(d):
    """DataFrame -> list of row dicts, NaN -> None (JSON-valid null)."""
    return d.astype(object).where(d.notna(), None).to_dict(orient="records")

inertia_true = float(np.sum(np.diag(Sigma_A)))

reference = {
    # --- genomic (X is the single source of truth for A_genomic) ---
    "index": list(pedigree["ID"]),
    "A": A.tolist(),
    "X": X.tolist(),
    "K_genomic": K.tolist(),   # frozen reference

    # --- simulation truths ---
    "p": p,
    "n_axes": 2,
    "means": m.tolist(),
    "Sigma_A": Sigma_A.tolist(),
    "Sigma_R": Sigma_R.tolist(),
    "Lambda": Lambda.tolist(),
    "Q": Q.tolist(),
    "inertia_total": inertia_true,
    "rel_inertia": [float(Lambda[a] / inertia_true) for a in range(p)],
    "u_true": u.tolist(),                 # (n, p)
    "var_additive_het": float(Sigma_A[0, 0]),
    "het_order": list(range(p)),

    # --- analysis tables ---
    "df_long": _df_records(df_long.assign(envt=df_long["envt"].astype(int))),
    "df_cols": _df_records(df_cols),
}

with open("../data/genomic_sim.json", "w") as f:
    json.dump(reference, f)

# %% [markdown]
# ## test varhet

# %%
mod_het = MixedModel.from_dataframe(
    data       = df_long,
    response   = "y",
    fixed      = "1 + envt",
    random = Random(
        unit         = "ID",
        right_hand   = "str",
        covariance   = K,
        matrix_index = list(pedigree["ID"]),
    ),
    residual = Residual(
        right_hand   = "het",
        het_formula  = "C(envt)"
    ),
).fit()

# %%
# ---------- environment means ----------
raw = mod_het.estimates["estimate"].tolist()
intercept = raw[0]
mean_hat = [intercept] + [intercept + c for c in raw[1:]]

print("=== Environment means ===")
for envt in range(p):
    print(f"  env {envt}: pyreml={mean_hat[envt]:.3f}  true={m[envt]:.3f}")
    
# ---------- additive variance ----------
var_a_hat = float(mod_het.random[0].build_S().detach())
print("\n=== Additive variance ===")
print(f"  pyreml={var_a_hat:.3f}  true (Sigma_A[0,0])={Sigma_A[0, 0]:.3f}")

# ---------- heterogeneous residual variances ----------
mod_het.residual.format_variance()
var = mod_het.residual.variance

sigma_base = float(var["sigma"])          # variance de la modalité de référence (env 0)
het = var["metadata"]["het"]              # [{column, h}, ...] pour les non-référence

# la référence (niveau patsy droppé = env 0) porte l'échelle ;
# les autres sont des ratios h à cette référence
sigma_r_hat = {0: sigma_base}
for entry in het:
    k = int(entry["column"].split("[T.")[1].rstrip("]"))   # "C(envt)[T.k]" -> k
    sigma_r_hat[k] = sigma_base * entry["h"]

print("\n=== Residual variances (het) ===")
print("  het formula:", var["metadata"]["het_formula"])
for envt in range(p):
    print(f"  env {envt}: pyreml={sigma_r_hat[envt]:.3f}  true={Sigma_R[envt]:.3f}")

# ---------- BLUP vs simulated u ----------
blup = mod_het.random[0].table[["unit", "prediction"]]
u_true = pd.DataFrame({"unit": pedigree["ID"].values, "u_true": u[:, 0]})
cmp = blup.merge(u_true, on="unit")

# restrict to phenotyped genotypes
obs_ids = set(df["ID"])

print("\n=== BLUP vs simulated u ===")
acc_all = np.corrcoef(cmp["prediction"], cmp["u_true"])[0, 1]
print(f"  accuracy (all levels):       {acc_all:.4f}")

# ---------- R² (y vs Xb + Zu) ----------
resid_tab = mod_het.residual.table          # columns: observation, response, residual
y_obs = df_long["y"].to_numpy()
e = resid_tab["residual"].to_numpy()         # y - (Xb + Zu)

ss_res = np.sum(e ** 2)
ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot
print(f"\n=== R² (y vs Xb + Zu) ===\n  R²={r2:.4f}")

# %%
reference_het = {
    "model": "het",
    "p": p,
    "id_index": list(pedigree["ID"]),
    "means": m.tolist(),                       # true environment means
    "var_additive": float(Sigma_A[0, 0]),      # true additive variance (env 0)
    "var_residual": Sigma_R.tolist(),          # true heterogeneous residual variances
    "het_order": list(range(p)),
    "u_true": u[:, 0].tolist(),                # shared additive effect (env 0)
}

with open("../data/genomic_het.json", "w") as f:
    json.dump(reference_het, f, indent=2)

# %% [markdown]
# ## test FA

# %%
mod_fa = MixedModel.from_dataframe(
    data     = df_cols,
    response = [f"y_{envt}" for envt in range(p)],
    fixed    = "1",
    random = Random(
        unit         = "ID",
        left_hand    = "fa",
        right_hand   = "str",
        covariance   = K,
        matrix_index = list(pedigree["ID"]),
        n_axes       = 2,
    ),
    residual = Residual(
        left_hand    = "diag"
    ),
).fit()

# %%
# ---------- environment means (one intercept per response) ----------
est = mod_fa.estimates                       # response | term | estimate
mean_fa = est[est["term"] == "Intercept"].sort_values("response")["estimate"].tolist()
print("=== Environment means (FA) ===")
for envt in range(p):
    print(f"  env {envt}: pyreml={mean_fa[envt]:.3f}  true={m[envt]:.3f}")

# ---------- factor-analytic structure ----------
fa = mod_fa.random[0].variance["metadata"]["fa"]
Q_hat = fa["Q"]                 # (p, 2), columns ordered by decreasing Lambda
Lambda_hat = fa["Lambda"]       # (2,)
S_hat = mod_fa.random[0].build_S().detach().numpy()
inertia_hat = np.trace(S_hat)   # total estimated inertia (low-rank + Psi)

# true side
inertia_true = np.sum(np.diag(Sigma_A))     # = sum(Lambda)

print("\n=== Factor axes: corr(estimated, simulated) ===")
for ax in range(2):
    r = np.corrcoef(Q_hat[:, ax], Q[:, ax])[0, 1]
    print(f"  axis {ax}: |corr|={abs(r):.4f}")

print("\n=== Relative inertia (axis i / total) ===")
for ax in range(2):
    rel_hat = Lambda_hat[ax] / inertia_hat
    rel_true = Lambda[ax] / inertia_true
    print(f"  axis {ax}: pyreml={rel_hat:.4f}  true={rel_true:.4f}")

# ---------- residual variances (diag) vs Sigma_R ----------
mod_fa.residual.format_variance()
S_r_hat = np.diag(mod_fa.residual.variance["sigma"])   # diag -> matrix; take diagonal
print("\n=== Residual variances (diag) ===")
for envt in range(p):
    print(f"  env {envt}: pyreml={S_r_hat[envt]:.3f}  true={Sigma_R[envt]:.3f}")

# ---------- BLUP vs simulated u (per environment) ----------
tab = mod_fa.random[0].table                 # unit | response | component | prediction
u_true_df = pd.DataFrame(
    {"unit": pedigree["ID"].values, **{f"y_{e}": u[:, e] for e in range(p)}}
)
print("\n=== BLUP vs simulated u (per environment) ===")
for envt in range(p):
    resp = f"y_{envt}"
    pred = tab[tab["response"] == resp][["unit", "prediction"]]
    cmp = pred.merge(u_true_df[["unit", resp]], on="unit")
    acc = np.corrcoef(cmp["prediction"], cmp[resp])[0, 1]
    print(f"  env {envt}: accuracy={acc:.4f}")

# ---------- R² (y vs Xb + Zu) ----------
resid_tab = mod_fa.residual.table
e = resid_tab["residual"].to_numpy()
y_stacked = np.concatenate(
    [df_cols[f"y_{e}"].dropna().to_numpy() for e in range(p)]
)
ss_res = np.sum(e ** 2)
ss_tot = np.sum((y_stacked - y_stacked.mean()) ** 2)
print(f"\n=== R² (y vs Xb + Zu) ===\n  R²={1 - ss_res/ss_tot:.4f}")

# %%
inertia_true = float(np.sum(np.diag(Sigma_A)))
reference_fa = {
    "model": "fa",
    "p": p,
    "n_axes": 2,
    "id_index": list(pedigree["ID"]),
    "means": m.tolist(),                       # true environment means
    "Sigma_A": Sigma_A.tolist(),               # true genetic covariance (full)
    "Lambda": Lambda.tolist(),                 # true eigenvalues (descending)
    "Q": Q.tolist(),                           # true eigenvectors (columns)
    "inertia_total": inertia_true,             # sum(diag(Sigma_A)) = sum(Lambda)
    "rel_inertia": [float(Lambda[a] / inertia_true) for a in range(p)],
    "var_residual": Sigma_R.tolist(),          # true residual variances (diag)
    "u_true": u.tolist(),                      # full additive effects (n x p)
}

with open("../data/genomic_fa.json", "w") as f:
    json.dump(reference_fa, f, indent=2)


