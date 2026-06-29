# %% [markdown]
# # Building tests

# %%
import numpy as np
import json
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from pprint import pprint

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
print(mod.random)
print(mod.response)
print(mod.fixed_names)
print(mod.AIC)
print(mod.AIC_meth)

print(mod.EEV[:5,:5])
print(mod.estimates[:5])
print(mod.residual.table[:5])

# %%
with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    df_r = ro.conversion.py2rpy(df)

mod_lm = stats.lm(stats.formula("height ~ 1 + BLOC + circumference"), data=df_r)

summ = ro.r("summary")(mod_lm)
coef_tab = np.array(ro.r("coef")(summ))  # (p, 4): Estimate, Std.Error, t value, Pr(>|t|)

result = {
    "beta": list(stats.coef(mod_lm)),
    "eev": np.array(ro.r("as.matrix")(stats.vcov(mod_lm))).tolist(),
    "residuals": list(stats.residuals(mod_lm)),
    "tvals": coef_tab[:, 2].tolist(),
    "aic": float(stats.AIC(mod_lm)[0]),
}
pprint(result)

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
print(mod.random)
print(mod.response)
print(mod.fixed_names)
print(mod.AIC)
print(mod.AIC_meth)

print(mod.EEV)
print(mod.estimates)
print(mod.residual.table[:5])

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
    coef_tab  = np.array(ro.r("coef")(ro.r("summary")(mod_lme4)))
    tvals     = coef_tab[:, 2].tolist()
    aic       = float(stats.AIC(mod_lme4)[0])

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
    "tvals": tvals,
    "aic": aic,
}

pprint(result)

with open("../data/regression_random.json", "w") as f:
    json.dump(result, f, indent=2)

# lme4 convergence
print(list(lme4.isSingular(mod_lme4)))
print(ro.r("summary")(mod_lme4).rx2("optinfo").rx2("conv").rx2("lme4"))



# %%
