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
# ## Simulate a genomic matrix from the pedigree

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
# ## Simulate a multi-trial scenario

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

sigma_base = float(var["sigma"])          # variance of the reference category (env 0)
het = var["metadata"]["het"]              # [{column, h}, ...] for non-reference categories

# the reference category (dropped Patsy level = env 0) carries the scale;
# the others are h ratios relative to this reference
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

# %% [markdown]
# ## Bivariate random regression — references bl_resp / bl_form / kr_resp / kr_form
# (reuses the beginning of the genomic script up to `K = K2`)

# %%
# d = k*c = 4, response-outer / formula-inner components:
#   0:(y_0,Intercept) 1:(y_0,x) 2:(y_1,Intercept) 3:(y_1,x)
corr = 0.5
var = 10.0
R = np.full((4, 4), corr)
np.fill_diagonal(R, 1.0)
Sigma_A = var * R                      # positive definite: eigenvalues 25 and 5 (x3)

Sigma_R = np.array([4.0, 6.0])         # diagonal residual across the 2 responses (distinct)

# true fixed effects (population): intercept and slope for each response
true_beta = {"y_0": (100.0, 3.0), "y_1": (50.0, -2.0)}

x_points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

print("Sigma_A =\n", Sigma_A)
print("Sigma_R =", Sigma_R)
print("true_beta =", true_beta)

# %%
# u ~ N(0, Sigma_A ⊗ K), using the same mechanism as in the previous script
id_to_row = {id_: i for i, id_ in enumerate(pedigree["ID"])}

Ls = np.linalg.cholesky(Sigma_A).T
Lg = np.linalg.cholesky(K).T
z = rng.normal(0, 1, size=4 * n).reshape(4, n)
u = (Ls @ z @ Lg.T).T                  # (n, 4), columns in component order
print("u.shape =", u.shape)

# %%
# records: each phenotyped genotype × each x value, with both responses observed
geno = df["ID"].drop_duplicates().to_numpy()
recs = []
for gid in geno:
    ri = id_to_row[gid]
    a0i, a0s, a1i, a1s = u[ri]          # (y0,Int)(y0,x)(y1,Int)(y1,x)
    for xv in x_points:
        gv0 = true_beta["y_0"][0] + true_beta["y_0"][1] * xv + a0i + a0s * xv
        gv1 = true_beta["y_1"][0] + true_beta["y_1"][1] * xv + a1i + a1s * xv
        y0 = gv0 + rng.normal(0, np.sqrt(Sigma_R[0]))
        y1 = gv1 + rng.normal(0, np.sqrt(Sigma_R[1]))
        recs.append((gid, xv, y0, y1))

df_cols = pd.DataFrame(recs, columns=["ID", "x", "y_0", "y_1"])
print(df_cols.head())
print("n_obs =", len(df_cols), " n_geno =", len(geno))

# %%
def fit_lh(left_hand):
    return MixedModel.from_dataframe(
        data     = df_cols,
        response = ["y_0", "y_1"],
        fixed    = "1 + x",
        random = Random(
            unit         = "ID",
            formula      = "1 + x",
            left_hand    = left_hand,
            right_hand   = "str",
            covariance   = K,
            matrix_index = list(pedigree["ID"]),
        ),
        residual = Residual(left_hand="diag"),
    ).fit()

models = {lh: fit_lh(lh) for lh in ["bl_resp", "bl_form", "kr_resp", "kr_form"]}

# %%
# block indices (response-outer / formula-inner)
RESP = {0: [0, 1], 1: [2, 3]}              # block / response
FORM = {"Intercept": [0, 2], "x": [1, 3]}  # block / formula term

def block(S, rows, cols):
    return S[np.ix_(rows, cols)]

# true sub-blocks of Sigma_A (what the nonzero blocks should approximate)
true_resp0 = block(Sigma_A, RESP[0], RESP[0])      # [[10,5],[5,10]]
true_form_int = block(Sigma_A, FORM["Intercept"], FORM["Intercept"])

for lh, mod in models.items():
    S = mod.random[0].build_S().detach().numpy()
    print(f"\n===== {lh} =====")
    print("S =\n", np.round(S, 3))

    if lh in ("bl_resp", "kr_resp"):
        cross = block(S, RESP[0], RESP[1])
        print("  ||cross-response block|| (expected 0) =", np.abs(cross).max())
        print("  response 0 block =\n", np.round(block(S, RESP[0], RESP[0]), 3))
        print("  response 1 block =\n", np.round(block(S, RESP[1], RESP[1]), 3))
        print("  true response block (reference) =\n", np.round(true_resp0, 3))
        if lh == "kr_resp":
            b0 = block(S, RESP[0], RESP[0])
            b1 = block(S, RESP[1], RESP[1])
            ratio = b1 / b0
            print("  response1/response0 ratio (expected ~constant, ~1 here) =\n", np.round(ratio, 4))

    if lh in ("bl_form", "kr_form"):
        cross = block(S, FORM["Intercept"], FORM["x"])
        print("  ||cross-formula block|| (expected 0) =", np.abs(cross).max())
        bi = block(S, FORM["Intercept"], FORM["Intercept"])
        bx = block(S, FORM["x"], FORM["x"])
        print("  Intercept block =\n", np.round(bi, 3))
        print("  x block =\n", np.round(bx, 3))
        print("  true formula block (reference) =\n", np.round(true_form_int, 3))
        if lh == "kr_form":
            print("  x/Intercept ratio (expected ~constant, ~1 here) =\n", np.round(bx / bi, 4))

    # diagonal residual
    mod.residual.format_variance()
    sr = np.diag(mod.residual.variance["sigma"])
    print("  estimated residual =", np.round(sr, 3), " true =", Sigma_R)

# %%
# performance: BLUP accuracy by (response, component) + R² by response
u_true_df = pd.DataFrame({
    "unit": pedigree["ID"].values,
    ("y_0", "Intercept"): u[:, 0],
    ("y_0", "x"):         u[:, 1],
    ("y_1", "Intercept"): u[:, 2],
    ("y_1", "x"):         u[:, 3],
})

for lh, mod in models.items():
    print(f"\n----- performance {lh} -----")
    tab = mod.random[0].table
    for resp in ["y_0", "y_1"]:
        for comp in ["Intercept", "x"]:
            pred = tab[(tab["response"] == resp) & (tab["component"] == comp)][["unit", "prediction"]]
            tru = u_true_df[["unit", (resp, comp)]].rename(columns={(resp, comp): "u_true"})
            cmp = pred.merge(tru, on="unit")
            acc = np.corrcoef(cmp["prediction"], cmp["u_true"])[0, 1]
            print(f"  acc {resp} {comp}: {acc:.4f}")

    resid_tab = mod.residual.table
    for resp in ["y_0", "y_1"]:
        e = resid_tab[resid_tab["response"] == resp]["residual"].to_numpy()
        yv = df_cols[resp].to_numpy()
        r2 = 1.0 - np.sum(e ** 2) / np.sum((yv - yv.mean()) ** 2)
        print(f"  R² {resp}: {r2:.4f}")

# %%
def _df_records(d):
    return d.astype(object).where(d.notna(), None).to_dict(orient="records")

reference_blkr = {
    "model": "bl_kr_randomreg",
    "k": 2, "c": 2, "d": 4,
    "components": [["y_0", "Intercept"], ["y_0", "x"],
                   ["y_1", "Intercept"], ["y_1", "x"]],
    "corr_target": corr,
    "var_target": var,
    "id_index": list(pedigree["ID"]),
    "Sigma_A": Sigma_A.tolist(),
    "Sigma_R": Sigma_R.tolist(),
    "true_beta": {r: list(v) for r, v in true_beta.items()},
    "x_points": x_points.tolist(),
    "u_true": u.tolist(),                # (n, 4), component order
    "df_cols": _df_records(df_cols),
}
with open("../data/genomic_blkr.json", "w") as f:
    json.dump(reference_blkr, f)
