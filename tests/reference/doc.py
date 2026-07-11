# %%
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import norm

from pyreml import (
    MixedModel,
    Random,
    Residual,
    A_pedigree,
    D_pedigree,
    A_genomic,
    prepare_pedigree,
    larix,
)

OUT = "../data"


def _df_records(d):
    """DataFrame -> list of row dicts, NaN -> None (JSON-valid null)."""
    return d.astype(object).where(d.notna(), None).to_dict(orient="records")


def _dump(obj, name):
    with open(f"{OUT}/doc_{name}.json", "w") as f:
        json.dump(obj, f, indent=2)

# %% [markdown]
# ## spatial : eucl random effect + kriging

# %%
df = larix.copy()
df = df[df["year"] == 2000].copy()

mod = MixedModel.from_dataframe(
    data     = df,
    response = "height",
    fixed    = "1",
    random   = Random(
        unit       = ["X", "Y"],
        right_hand = "eucl",
    ),
).fit()

coords = df[["X", "Y"]].to_numpy()
train_coords = [tuple(c) for c in coords]


def _kriging_grid(step):
    gx = np.arange(
        coords[:, 0].min() - 20,
        coords[:, 0].max() + 20,
        step,
    )
    gy = np.arange(
        coords[:, 1].min() - 20,
        coords[:, 1].max() + 20,
        step,
    )

    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])

    train_blup = {
        (x, y): u
        for (x, y), u in zip(
            train_coords,
            mod.random[0].table["prediction"],
        )
    }

    new_cells = [
        tuple(c)
        for c in grid
        if tuple(c) not in train_blup
    ]

    prediction_df = mod.random[0].predict(
        matrix_index=train_coords + new_cells
    )

    pred_blup = {
        (row["X"], row["Y"]): row["prediction"]
        for _, row in prediction_df.iterrows()
    }

    surface = np.array(
        [
            train_blup.get(tuple(c), pred_blup.get(tuple(c)))
            for c in grid
        ],
        dtype=float,
    ).reshape(GX.shape)

    return gx, gy, GX, grid, prediction_df, surface


# coarse grid (step 10): frozen reference -> JSON
gx, gy, GX, grid, prediction_df, surface = _kriging_grid(step=10)

gx, gy, GX, grid, prediction_df, surface = _kriging_grid(step=10)

reference_spatial = {
    "series": "spatial",
    "n_train": int(len(df)),
    "n_grid": int(len(grid)),
    "estimates": _df_records(mod.estimates),
    "rho": float(mod.random[0].variance["metadata"]["rho"]),
    "var_additive": float(mod.random[0].build_S().detach()),
    "blup": _df_records(mod.random[0].table),
    "residuals": _df_records(mod.residual.table),
    "prediction": _df_records(prediction_df),
}

_dump(reference_spatial, "spatial")

# fine grid (step 1): figure only, not stored
gx, gy, GX, grid, prediction_df, surface = _kriging_grid(step=1)
m = np.nanmax(np.abs(surface))

plt.imshow(
    surface,
    origin="lower",
    extent=[gx.min(), gx.max(), gy.min(), gy.max()],
    aspect="equal",
    cmap="RdBu_r",
    vmin=-m,
    vmax=m,
)
plt.colorbar(label="prediction")
plt.scatter(df["X"], df["Y"], c="black", s=8)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# %% [markdown]
# ## regression : random regression (1 + year), full left-hand factor

# %%
df = larix.copy()
df["year"] = df["year"] - df["year"].min()

ped = prepare_pedigree(df[["ID", "DAM", "SIRE"]])
K = A_pedigree(ped)

mod = MixedModel.from_dataframe(
    data     = df,
    response = "height",
    fixed    = "1 + year",
    random   = Random(
        unit         = "ID",
        formula      = "1 + year",
        left_hand    = "full",
        right_hand   = "str",
        covariance   = K,
        matrix_index = ped["id"].tolist(),
        jitter       = 1e-6,
    ),
    residual = Residual(
        left_hand = "iid",
        right_hand = "het",
        het_formula = "C(year)"
    ),
).fit()

reg_var = mod.random[0].variance

reference_regression = {
    "series": "regression",
    "estimates": _df_records(mod.estimates),
    "S_random": np.asarray(reg_var["sigma"]).tolist(),
    "variance_labels": reg_var["metadata"]["labels"],
    "blup": _df_records(mod.random[0].table),
    "residuals": _df_records(mod.residual.table),
}
_dump(reference_regression, "regression")

# figure: individual growth curves
b = mod.estimates.set_index("term")["estimate"]
b0, b1 = b["Intercept"], b["year"]

re = mod.random[0].table.pivot_table(index="unit", columns="component", values="prediction")

yr = np.linspace(df["year"].min(), df["year"].max(), 100)

fig, ax = plt.subplots(figsize=(7, 5))
for _, row in re.iterrows():
    a0, a1 = row["Intercept"], row["year"]
    ax.plot(yr, (b0 + a0) + (b1 + a1) * yr,
            color="#4B8BBE", lw=0.6, alpha=0.15)
ax.plot(yr, b0 + b1 * yr, color="black", lw=1, label="Population mean")
plt.title("Individual growth curves")
ax.set_xlabel("year"); ax.set_ylabel("height")
ax.legend()
plt.show()


# %% [markdown]
# ## kinships : A / D (pedigree) and G (genomic) -- figures only, no JSON

# %%
df = larix.copy()
ped = prepare_pedigree(df[["ID", "SIRE", "DAM"]])

A = A_pedigree(ped)
D = D_pedigree(ped)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(A, cmap="coolwarm", center=0, ax=axes[0], square=True)
axes[0].set_title("A (additive)")
sns.heatmap(D, cmap="coolwarm", center=0, ax=axes[1], square=True)
axes[1].set_title("D (dominance)")
plt.tight_layout()
plt.show()

# genomic matrix simulated from the pedigree kinship (deterministic, seed 42)
rng = np.random.default_rng(42)
n = A.shape[0]
m = 10_000
maf = rng.uniform(0.05, 0.95, size=m)
thr = norm.ppf(1 - maf)
L = np.linalg.cholesky(A)
X = np.zeros((n, m))
for l in range(m):
    g1 = (L @ rng.standard_normal(n)) > thr[l]
    g2 = (L @ rng.standard_normal(n)) > thr[l]
    X[:, l] = (g1.astype(float) - 0.5) + (g2.astype(float) - 0.5)

G = A_genomic(
    X,
    min_MAF     = 0.05,
    max_missing = 0.1,
    shrink      = True,
)

sns.heatmap(G, cmap="coolwarm", center=0, square=True)
plt.title("G (additive, genomic)")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## fa : multi-trait factor-analytic model

# %%
df = larix.copy()
traits = ["height", "circumference", "flexuosity"]
df = df[df["year"].isin([2000, 2014])]
df[traits] = (df[traits] - df[traits].mean()) / df[traits].std()

long = df.melt(
    id_vars=["ID", "DAM", "SIRE", "BLOC", "year"],
    value_vars=traits,
    var_name="trait",
    value_name="value",
)
long["resp"] = long["trait"] + "_" + long["year"].astype(str)
wide = (
    long.pivot_table(
        index=["ID", "DAM", "SIRE", "BLOC"],
        columns="resp",
        values="value",
    )
    .dropna(axis=1, how="all")
    .reset_index()
)
responses = [c for c in wide.columns if c not in ("ID", "DAM", "SIRE", "BLOC")]

ped = prepare_pedigree(wide[["ID", "DAM", "SIRE"]])
A = A_pedigree(ped)

P = wide[responses].cov().to_numpy()

model = MixedModel.from_dataframe(
    data     = wide,
    response = responses,
    fixed    = "1",
    random   = Random(
        unit         = "ID",
        left_hand    = "fa",
        n_axes       = 2,
        right_hand   = "str",
        covariance   = A,
        matrix_index = ped["id"].tolist(),
        init         = P / 2,
        jitter       = 1e-6,
    ),
    residual = Residual(
        left_hand    = "fa",
        n_axes       = 2,
        right_hand   = "iid",
        init         = P / 2,
        jitter       = 1e-6,
    ),
    device = "cuda",
).fit()

ran_var = model.random[0].variance
res_var = model.residual.variance
ran_fa = ran_var["metadata"]["fa"]
res_fa = res_var["metadata"]["fa"]

reference_fa = {
    "series": "fa",
    "responses": list(responses),
    "estimates": _df_records(model.estimates),

    # genetic (random) FA structure
    "S_random": np.asarray(ran_var["sigma"]).tolist(),
    "Q_random": np.asarray(ran_fa["Q"]).tolist(),
    "Lambda_random": np.asarray(ran_fa["Lambda"]).tolist(),
    "Psi_random": np.asarray(ran_fa["Psi"]).tolist(),

    # residual FA structure
    "S_residual": np.asarray(res_var["sigma"]).tolist(),
    "Q_residual": np.asarray(res_fa["Q"]).tolist(),
    "Lambda_residual": np.asarray(res_fa["Lambda"]).tolist(),
    "Psi_residual": np.asarray(res_fa["Psi"]).tolist(),

    "blup": _df_records(model.random[0].table),
    "residuals": _df_records(model.residual.table),
}
_dump(reference_fa, "fa")

# figure: correlation circle + average-linkage clustering of the genetic FA
S  = np.asarray(ran_var["sigma"])
Q, Lam, Psi = np.asarray(ran_fa["Q"]), np.asarray(ran_fa["Lambda"]), np.asarray(ran_fa["Psi"])
Gamma = Q * np.sqrt(Lam)

Dn   = inv(sqrtm(np.diag(np.diag(S))))
cor  = Dn @ S @ Dn
dist = 1 - cor
np.fill_diagonal(dist, 0)
Z = linkage(squareform(np.round(dist, 10)), method="average")

PC   = Dn @ Gamma
expl = Lam / np.trace(S)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.add_patch(plt.Circle((0, 0), 1, fill=False, color="grey", lw=0.8))
ax1.axhline(0, color="grey", lw=0.5); ax1.axvline(0, color="grey", lw=0.5)
for i, name in enumerate(responses):
    ax1.annotate("", xy=(PC[i, 0], PC[i, 1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="steelblue"))
    ax1.text(PC[i, 0] * 1.05, PC[i, 1] * 1.05, name, fontsize=7)
ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1); ax1.set_aspect("equal")
ax1.set_xlabel(f"Axis 1 ({100 * expl[0]:.0f}%)")
ax1.set_ylabel(f"Axis 2 ({100 * expl[1]:.0f}%)")

labels = [r.replace("_", "\n") for r in responses]
dendrogram(Z, labels=labels, ax=ax2, color_threshold=0)
ax2.set_ylabel("Distance")
ax2.tick_params(axis="x", labelsize=9)
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

fig.tight_layout()
plt.show()

# %%
