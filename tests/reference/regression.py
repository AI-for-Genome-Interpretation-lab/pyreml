# %% [markdown]
# # Building tests

# %%
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

from pyreml import MixedModel, Random, larix as df

lme4 = importr("lme4")
stats = importr("stats")
base = importr("base")
nlme = importr("nlme")
fixef  = ro.r["fixef"]
ranef  = ro.r["ranef"]
VarCorr = ro.r["VarCorr"]

df = df[df["year"] == 2000]
df["circumference"] = (df["circumference"] - df["circumference"].mean()) / df["circumference"].std()
df["height"] = (df["height"] - df["height"].mean()) / df["height"].std()

df.head()

# %%
mod = MixedModel.from_dataframe(
    data = df,
    response = "height",
    fixed = "1 + BLOC + circumference",
).fit()

print(mod.do_REML)
print(mod.do_HMME)
print(mod.random)
print(mod.response)
print(mod.fixed_names)

print(mod.EEV)
print(mod.estimates)
print(mod.residual.table)

# %%
ref = smf.ols("height ~ 1 + BLOC + circumference", data=df).fit()

result = {
    "beta": ref.params.tolist(),
    "eev": ref.cov_params().values.tolist(),
    "residuals": ref.resid.tolist(),
}

with open("../data/regression_fixed.json", "w") as f:
    json.dump(result, f, indent=2)

# %%
mod = MixedModel.from_dataframe(
    data = df,
    response = "height",
    fixed = "1 + circumference",
    random = Random(
        unit = "BLOC",
        formula = "1 + circumference",
        left_hand= "full",
    )
).fit()

print(mod.do_REML)
print(mod.do_HMME)
print(mod.random)
print(mod.response)
print(mod.fixed_names)

print(mod.EEV)
print(mod.estimates)
print(mod.residual.table)

# %%
with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    df_r = ro.conversion.py2rpy(df)

mod_lme4= lme4.lmer(
    stats.formula('height ~ 1 + circumference + (1 + circumference | BLOC)'),
    REML = True,
    data = df_r
)

with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    beta      = list(fixef(mod_lme4))
    eev       = np.array(ro.r("as.matrix")(stats.vcov(mod_lme4))).tolist()
    varcorr   = np.array(VarCorr(mod_lme4)[0]).tolist()
    sigma_r   = float(stats.sigma(mod_lme4)[0]) ** 2
    blup      = np.array(ranef(mod_lme4)[0]).tolist()
    residuals = list(stats.residuals(mod_lme4))

ranef_cv = ranef(mod_lme4, **{"condVar": True})
pev_raw = np.array(ro.r("attr")(ranef_cv[0], "postVar"))
pev = pev_raw.tolist()

result = {
    "beta": beta,
    "eev": eev,
    "varcorr": varcorr,
    "sigma_r": sigma_r,
    "blup": blup,
    "residuals": residuals,
    "pev": pev,
}

with open("../data/regression_random.json", "w") as f:
    json.dump(result, f, indent=2)

# lme4 convergence
print(list(lme4.isSingular(mod_lme4)))
print(ro.r("summary")(mod_lme4).rx2("optinfo").rx2("conv").rx2("lme4"))


