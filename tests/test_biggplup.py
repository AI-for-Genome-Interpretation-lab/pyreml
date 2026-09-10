"""
Big-GBLUP: the factored path must stay factored, and must still recover.

Two assertions live here, and they are independent.

DIMENSIONAL. On the factored path every legitimate allocation falls in
max(q^2, n*p). A dense incidence is n*q and a dense kernel or gradient matrix is
n^2. A TorchDispatchMode records the largest tensor any aten op produced during
construction AND fit, and the ceiling is derived from the model's own
dimensions rather than measured, so it is independent of dtype, device and
machine, and it fires at any scale. A byte threshold would not: at this size a
dense n x n residual grain is 4 GiB -- enough to break a GPU run, small enough
to slip under a generous RAM ceiling, invisible at a quarter of the scale.

The probe sits at the dispatcher, so it has no list of watched objects: a
regression moving the bulk onto a different attribute, a local, or an anonymous
temporary is seen just the same. Its blind spot is numpy and scipy, which is
where the dense-Z route allocates (Random.Z, scipy.linalg.block_diag, np.hstack)
-- hence the routing assertions below, which name the gardes directly.

RECOVERY. The reference states what the simulation put in and how far the
estimates may sit from it. It does not freeze a previous run, so an
eighth-decimal numerical drift passes here; that detection lives in
test_genomics and test_pedigree, which carry R references. What this catches is
rupture -- a component that stops being identified at all.

One dataset, two models. `het` is the generating model. `iid` is deliberately
misspecified (random environment, homoscedastic residual on heteroscedastic
data) and is here for its TWO random blocks, which exercise the nested loops of
the factored incidence facing -- degenerate with a single block. Its targets are
the marginal quantities the misspecification leaves identifiable.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from pyreml import MixedModel, Random, Residual, A_genomic

DEVICE = "cpu"
DTYPE = "double" # do not change

HERE = os.path.dirname(os.path.abspath(__file__))
GENO = os.path.join(HERE, "data", "arabido.npz")
REF_JSON = os.path.join(HERE, "data", "bigglup.json")
REF_NPZ = os.path.join(HERE, "data", "bigglup.npz")

SCENARIOS = ["het", "iid"]

# Multiple of max(q^2, n*p) the largest allocation may reach. Structural, not
# measured: views (reshape, expand) inflate numel without allocating, so the
# margin absorbs them while staying far below n*q. Kept here rather than in the
# reference because it is a property of the model dimensions, not of the draw.
NUMEL_MARGIN = 4

# The gardes that must resolve for the factored path to be taken. Asserted by
# name so that a silent fallback to a dense route is reported as such, rather
# than as an unexplained ceiling breach.
ROUTING = {
    "SMW": True,
    "structured_forward": True,
    "analytic_backward": True,
}


class NumelProbe(TorchDispatchMode):
    """Largest tensor produced by any aten op, and the op that produced it.

    The witness is kept because the number alone does not say whether a breach
    is a dense incidence or a dense kernel.
    """

    def __init__(self):
        super().__init__()
        self.max_numel = 0
        self.witness = ""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(out):
            if isinstance(leaf, torch.Tensor) and leaf.numel() > self.max_numel:
                self.max_numel = leaf.numel()
                self.witness = str(func)
        return out


@pytest.fixture(scope="session")
def ref():
    with open(REF_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def data():
    """Kinship, phenotypes and truth, shared by both scenarios.

    A_genomic is the expensive step (two n^2 * m products under shrinkage), so
    it runs once per session. It writes into its argument, hence the fresh cast.
    """
    with np.load(GENO) as z:
        snp = z["snp"].astype(np.float64) - 1.0

    with np.load(REF_NPZ) as z:
        individual = z["individual"]
        envt = z["envt"]
        y = z["y"]
        u_true = z["u_true"]
        level_index = z["level_index"]

    K = A_genomic(snp, min_MAF=0.05, max_missing=0.1, shrink=True)
    del snp

    df = pd.DataFrame({"individual": individual, "envt": envt, "y": y})

    return {
        "K": K,
        "df": df,
        "u_true": u_true,
        "level_index": level_index,
    }


@pytest.fixture(scope="session", params=SCENARIOS)
def fitted(request, ref, data):
    """Fit one scenario under the probe, and hand back the model with it.

    Construction is inside the `with`, not only the fit: the dense-Z route
    allocates in from_dataframe, so a probe wrapping fit() alone would miss it.
    """
    scenario = request.param
    labels = ref["config"]["env_labels"]

    df = data["df"].copy()
    df["envt"] = pd.Categorical(df["envt"], categories=labels)

    geno = Random(
        formula="1",
        unit="individual",
        right_hand="str",
        covariance=data["K"],
        matrix_index=list(data["level_index"]),
    )

    if scenario == "het":
        fixed = "1 + C(envt)"
        random = geno
        residual = Residual(right_hand="het", het_formula="1 + C(envt)")
    else:
        fixed = "1"
        random = [geno, Random(formula="1", unit="envt", right_hand="iid")]
        residual = Residual()

    probe = NumelProbe()
    with probe:
        model = MixedModel.from_dataframe(
            data=df,
            response="y",
            fixed=fixed,
            random=random,
            residual=residual,
            device=DEVICE,
        ).fit(DTYPE, verbose=False)

    return {"scenario": scenario, "model": model, "probe": probe}


def _bracket(name, value, target, tol):
    lo, hi = tol
    ratio = value / target
    assert lo <= ratio <= hi, (
        f"{name}: {value:.4f} against a target of {target:.4f} "
        f"(ratio {ratio:.3f}, allowed {lo}-{hi})"
    )


# ---- dimensional -----------------------------------------------------------

def test_routing(fitted):
    """The gardes resolved onto the factored path."""
    model = fitted["model"]

    for flag, expected in ROUTING.items():
        assert getattr(model, flag) is expected, f"{flag} resolved the wrong way"

    # the dense incidence was never materialized, neither by the model nor by
    # the residual embedding
    assert model._Z is None
    assert model.variance.embed is not None
    assert model.variance.embed.Zg is None

    # the residual is a diagonal selector, which is what lets every Z product
    # run through the per-block facing instead of a dense Rinv
    assert model.residual.R_is_diagonal
    assert model.residual.W_is_identity


def test_dimensions(fitted, ref):
    """n, p and q are what the design implies, so the ceiling means something."""
    model = fitted["model"]
    cfg = ref["config"]
    n_ind, n_env = cfg["n_ind"], cfg["n_env"]

    assert model.n == cfg["n_obs"]

    if fitted["scenario"] == "het":
        assert model.p == n_env          # intercept + n_env - 1 contrasts
        assert model.q == n_ind          # one random block
        assert model.df_var == 2 + n_env - 1
    else:
        assert model.p == 1
        assert model.q == n_ind + n_env  # two random blocks
        assert model.df_var == 3


def test_no_quadratic_allocation(fitted):
    """Nothing allocated may leave the max(q^2, n*p) class.

    The three size classes must stay separated by orders of magnitude for the
    assertion to be worth anything, so that separation is checked first: a
    reference that let n*q through would pass silently otherwise.
    """
    model, probe = fitted["model"], fitted["probe"]
    n, p, q = model.n, model.p, model.q

    legit = max(q * q, n * p)
    cap = NUMEL_MARGIN * legit

    assert cap < n * q, (
        f"the cap ({cap:,}) does not exclude a dense incidence "
        f"(n*q = {n * q:,}); the test is vacuous as parameterized"
    )

    assert probe.max_numel <= cap, (
        f"largest allocation {probe.max_numel:,} exceeds the cap {cap:,} "
        f"[legit max(q^2, n*p) = {legit:,}, n*q = {n * q:,}, n^2 = {n * n:,}]. "
        f"Produced by {probe.witness}"
    )


# ---- recovery --------------------------------------------------------------

def test_genetic_variance(fitted, ref):
    scenario = fitted["scenario"]
    tol = ref["tol"][scenario]
    target = ref["targets"][scenario]["sigma_g"]
    sigma_g = float(fitted["model"].random[0].variance["sigma"])

    _bracket("sigma_g", sigma_g, target, tol["sigma_g"])


def test_residual_variance(fitted, ref):
    """The residual scale.

    In `het` this is the reference environment's variance; in `iid` it is the
    mean of the per-environment variances, which is where a single scale lands
    on a balanced heteroscedastic design.
    """
    scenario = fitted["scenario"]
    tol = ref["tol"][scenario]
    target = ref["targets"][scenario]["sigma_r"]
    sigma_r = float(fitted["model"].residual.variance["sigma"])

    _bracket("sigma_r", sigma_r, target, tol["sigma_r"])


def test_environment_variances(fitted, ref):
    """het: the per-environment residual variances.

    Only the PRODUCT sigma_r * h[e] is identified -- the scale and the ratios
    trade against each other -- so the product is what is asserted, never h
    alone.
    """
    if fitted["scenario"] != "het":
        pytest.skip("het scenario only")

    model = fitted["model"]
    tol = ref["tol"]["het"]["env_var"]
    targets = ref["targets"]["het"]["env_var"]

    sigma_r = float(model.residual.variance["sigma"])
    h = {d["column"]: float(d["h"])
         for d in model.residual.variance["metadata"]["het"]}

    assert set(h) == set(targets), "het columns do not match the reference"

    for term, target in targets.items():
        _bracket(f"env_var[{term}]", sigma_r * h[term], target, tol)


def test_environment_variance_component(fitted, ref):
    """iid: the variance of the random environment effect.

    The target is the REALIZED variance of the 20 draws, not the parametric one
    they came from: REML estimates the variance of the sample it sees, and at 20
    levels the two are far apart. The bracket is wide accordingly -- this
    component is a smoke check, not a measurement.
    """
    if fitted["scenario"] != "iid":
        pytest.skip("iid scenario only")

    model = fitted["model"]
    target = ref["targets"]["iid"]["sigma_env"]
    sigma_env = float(model.random[1].variance["sigma"])

    _bracket("sigma_env", sigma_env, target, ref["tol"]["iid"]["sigma_env"])


def test_fixed_effects(fitted, ref):
    """Additive tolerance: a treatment contrast may legitimately be near zero."""
    scenario = fitted["scenario"]
    atol = ref["tol"][scenario]["fixed_atol"]
    targets = ref["targets"][scenario]["fixed"]

    est = fitted["model"].estimates.set_index("term")["estimate"]

    for term, target in targets.items():
        assert term in est.index, f"missing fixed term {term}"
        dev = abs(float(est[term]) - target)
        assert dev <= atol, (
            f"{term}: {float(est[term]):.3f} against a target of {target:.3f} "
            f"(deviation {dev:.3f}, allowed {atol})"
        )


def test_blup_accuracy(fitted, ref, data):
    """A lower bound, not a target: saturated at this number of records."""
    tab = fitted["model"].random[0].table.set_index("unit")
    u_hat = tab.loc[data["level_index"], "prediction"].to_numpy()

    acc = float(np.corrcoef(u_hat, data["u_true"])[0, 1])
    assert acc >= ref["blup_acc_min"], (
        f"BLUP accuracy {acc:.4f} below the floor {ref['blup_acc_min']}"
    )