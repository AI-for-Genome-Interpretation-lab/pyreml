"""
Freeze the reference for the big-GBLUP dimensional test.

This script fits NOTHING. It simulates one dataset and states what a solver is
expected to recover from it, and within which margin. The targets are therefore
properties of the simulation, not a snapshot of a run: a regression is caught
because pyreml stops recovering what was simulated, not because it stops
reproducing what it produced yesterday.

The trade-off is explicit. A frozen -2logLik would catch an eighth-decimal
numerical drift; simulation targets will not. That fine-grained detection lives
in test_genomics and test_pedigree, which carry R references. What this catches
instead is rupture: a variance component that stops being identified, a routing
garde that stops resolving, an allocation that becomes quadratic.

ONE dataset, read by TWO models. The phenotype is simulated heteroscedastic,
with a fixed mean per environment; the two scenarios then differ only in how
they read it:

    het  environment FIXED, heteroscedastic residual. The generating model, so
         every target below is met head on. One random block, and a het residual
         whose K runs at the observation granularity (L_resid = n) -- the
         configuration that used to allocate an n x n gradient grain.

    iid  environment RANDOM, homoscedastic residual. DELIBERATELY misspecified:
         the environment means become a random sample of size N_ENV, and the
         residual becomes a single scale where the truth has N_ENV of them. It
         is not there to be exact, it is there because it carries TWO random
         blocks, which is what exercises the nested loops of the factored
         incidence facing (ZtWZ, _diag_ZCinvZ, Zv, ZtM) -- degenerate with a
         single block. Its targets are the marginal quantities the
         misspecification leaves identifiable, with brackets to match.

The observation-level arrays go to a sibling .npz rather than into the JSON:
indenting 22500 records would make the reference unreadable, which is the exact
opposite of the point. The JSON holds the config, the targets and the
tolerances, and stays a file a human can open and argue with.

The complexity ceilings are NOT stored here. They are structural properties of
the model dimensions, not of the draw, and belong in the test where they can be
read against max(q^2, n*p), n*q and n^2 in one glance.

Run from tests/reference/. Writes ../data/bigglup.json and ../data/bigglup.npz.
"""

# %% load packages
import json

import numpy as np
import pandas as pd

from pyreml import A_genomic

# %% config
SEED = 42

GENO = "../data/arabido.npz"
OUT_JSON = "../data/bigglup.json"
OUT_NPZ = "../data/bigglup.npz"

N_ENV = 20                  # environments, every line recorded once in each

SIGMA_G = 10.0              # additive genetic variance
SIGMA_R = 10.0              # residual variance IN THE REFERENCE ENVIRONMENT
BETA = 100.0                # grand mean

ENV_MEAN_SD = 15.0          # sd of the environment means
ENV_LOGRATIO_SD = 0.5       # sd of the residual log-variance ratios

# Tolerances, as multiplicative brackets on the estimate / target ratio, except
# fixed_atol which is additive because a contrast may legitimately be near zero.
#
# Wide on purpose, and each sized on what the SAMPLE can say rather than on what
# the solver can do:
#   - sigma_g rests on N_IND levels with N_ENV records each: tight.
#   - het.sigma_r and het.env_var rest on ~N_IND records per environment, but
#     the scale / ratio split is only identified through their PRODUCT, so the
#     product carries the tighter bracket and the scale a looser one.
#   - iid.sigma_env rests on N_ENV = 20 levels. Its relative standard error is
#     at best sqrt(2/20) ~ 32%, so a tight bracket would flap. Raise N_ENV to
#     50-100 if that component is to become informative.
#   - iid.sigma_r absorbs a heteroscedasticity the model cannot express, so its
#     bracket is the loosest of all: it says "the marginal scale is recovered",
#     not "the residual structure is".
TOL = {
    "het": {
        "sigma_g": (0.80, 1.25),
        "sigma_r": (0.70, 1.45),
        "env_var": (0.75, 1.35),    # sigma_r * h[e], the identified quantity
        "fixed_atol": 2.0,
    },
    "iid": {
        "sigma_g": (0.75, 1.30),    # looser: the residual misfit leaks here
        "sigma_r": (0.60, 1.60),
        "sigma_env": (0.30, 3.00),  # 20 levels: uninformative by construction
        "fixed_atol": 8.0,          # the intercept now carries the env sample
    },
}
BLUP_ACC_MIN = 0.95         # lower bound, not a target: saturated at N_ENV reps


# %% genotypes and kinship
with np.load(GENO) as z:
    # A_genomic writes into its argument (it overwrites monomorphic columns in
    # place), so this cast must stay a fresh array
    snp = z["snp"].astype(np.float64) - 1.0
    individuals = z["individual"]

K = A_genomic(snp, min_MAF=0.05, max_missing=0.1, shrink=True)
del snp

N_IND = len(individuals)
N_OBS = N_IND * N_ENV
print(f"{N_IND} lines, {N_ENV} environments, {N_OBS} records")


# %% simulate
rng = np.random.default_rng(np.random.SeedSequence([SEED, N_IND, N_ENV]))

# Breeding values are drawn THROUGH the kinship the solver will be given, not
# white: the model fits u ~ N(0, sigma_g * K), so a white draw would leave
# sigma_g unidentified and no target could be stated for it.
Lk = np.linalg.cholesky(K)
u = np.sqrt(SIGMA_G) * (Lk @ rng.standard_normal(N_IND))
del Lk, K

env_mean = rng.normal(scale=ENV_MEAN_SD, size=N_ENV)
env_logratio = rng.normal(scale=ENV_LOGRATIO_SD, size=N_ENV)
env_logratio[0] = 0.0                      # environment 0 is the reference
env_var = SIGMA_R * np.exp(env_logratio)   # residual variance of each env

# balanced crossed design, then shuffled so that no structure survives in the
# stacking order
ind_index = np.repeat(np.arange(N_IND), N_ENV)
env_index = np.tile(np.arange(N_ENV), N_IND)
order = rng.permutation(N_OBS)
ind_index, env_index = ind_index[order], env_index[order]

# zero-padded labels: patsy sorts the categories, so the lexicographic order is
# the numeric one and env_00 is the treatment-coding reference
env_labels = [f"env_{e:02d}" for e in range(N_ENV)]

df = pd.DataFrame({
    "individual": individuals[ind_index],
    "envt": pd.Categorical([env_labels[e] for e in env_index],
                           categories=env_labels),
    "y": (
        BETA + env_mean[env_index] + u[ind_index]
        + rng.normal(size=N_OBS) * np.sqrt(env_var[env_index])
    ),
})


# %% targets
# het reads the generating model, so its targets are the parameters themselves,
# re-expressed in patsy's treatment coding: the Intercept carries
# BETA + env_mean[0] and each dummy the contrast against environment 0.
het_fixed = {"Intercept": BETA + env_mean[0]}
het_fixed.update({
    f"C(envt)[T.{lab}]": env_mean[e] - env_mean[0]
    for e, lab in enumerate(env_labels) if e > 0
})

# Only the PRODUCT sigma_r * h[e] is identified: the scale and the ratios trade
# against each other, so the product carries the target and h alone does not.
het_env_var = {
    f"C(envt)[T.{lab}]": float(env_var[e])
    for e, lab in enumerate(env_labels) if e > 0
}

# iid reads a model the data were not generated under, so its targets are the
# marginal quantities that survive the misspecification:
#   - the environment effects become a sample of N_ENV draws, and REML estimates
#     the variance of that SAMPLE, which at 20 levels is far from ENV_MEAN_SD^2;
#     the realized variance is therefore the honest target, not the parametric
#     one it was drawn from.
#   - a single residual scale fitted on a balanced heteroscedastic design lands
#     on the mean of the per-environment variances.
#   - the intercept becomes the grand mean over the environment sample.
iid_targets = {
    "sigma_g": SIGMA_G,
    "sigma_r": float(env_var.mean()),
    "sigma_env": float(env_mean.var(ddof=1)),
    "fixed": {"Intercept": BETA + float(env_mean.mean())},
}

reference = {
    "config": {
        "seed": SEED,
        "n_ind": N_IND, "n_env": N_ENV, "n_obs": N_OBS,
        "sigma_g": SIGMA_G, "sigma_r": SIGMA_R, "beta": BETA,
        "env_mean_sd": ENV_MEAN_SD, "env_logratio_sd": ENV_LOGRATIO_SD,
        "env_labels": env_labels,
    },
    "targets": {
        "het": {
            "sigma_g": SIGMA_G,
            "sigma_r": SIGMA_R,
            "env_var": het_env_var,
            "fixed": het_fixed,
        },
        "iid": iid_targets,
    },
    "tol": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                for kk, vv in v.items()}
            for k, v in TOL.items()},
    "blup_acc_min": BLUP_ACC_MIN,
    "truth": {
        "env_mean": env_mean.tolist(),
        "env_var": env_var.tolist(),
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(reference, f, indent=2)

# observation-level arrays: too long to indent, and nothing a reader would open
np.savez_compressed(
    OUT_NPZ,
    individual=df["individual"].to_numpy().astype("U"),
    envt=df["envt"].astype(str).to_numpy().astype("U"),
    y=df["y"].to_numpy(),
    u_true=u,
    level_index=individuals,
)
