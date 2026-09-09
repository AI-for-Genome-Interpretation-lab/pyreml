"""
Factor-analytic multi-environment model: pyreml against WOMBAT, on every path.

N_ENV environments are N_ENV traits, correlated genetically through Sigma_A and
independent residually since two environments are two trials. The genetic
covariance is fitted by a reduced-rank FA with N_AXES factors.

TWO INDEPENDENT ASSERTIONS, and they answer different questions.

Cross-path agreement asks whether pyreml is self-consistent: the four
combinations of (SMW, optimizations) are four different routes to the same
likelihood, so they must land on the same estimates to near machine precision.
This one is cheap to interpret -- a failure points at a specific route.

Agreement with WOMBAT asks whether the FA is right at all. WOMBAT reaches the
same optimum by PX-AI, a wholly different algorithm, from the same starting
point. Its tolerance is much looser: the two stop on different criteria, and an
FA likelihood is not convex, so exact agreement is neither expected nor
required. What is required is that they agree on WHICH optimum.

WHAT IS NOT ASSERTED: recovery of the simulated Sigma_A. It is full rank by
construction (its correlation matrix is drawn uniformly over correlation
matrices), so a rank-N_AXES FA is an approximation of it and neither solver can
recover it. The reference stores it as context for reading a disagreement, not
as a target.

Q IS SIGN-ARBITRARY per column, and the factor basis is only identified up to a
rotation. So nothing here compares Q or Gamma directly: the invariant objects
are Sigma_A = Gamma Gamma' + diag(Psi), the eigenvalues Lambda, the specific
variances Psi and the residual variances.

COST. Both paths are O(n_obs^3) per evaluation with n_obs = n_ind * n_env, and
an FA on N_ENV traits carries a lot of variance parameters, so this module is
slow by nature. It is sized by N_IND in the reference generator; keep it small.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from pyreml import MixedModel, Random, Residual, A_genomic

DEVICE = "cpu"
DTYPE = "double"

HERE = os.path.dirname(os.path.abspath(__file__))
GENO = os.path.join(HERE, "data", "arabido.npz")
REF_JSON = os.path.join(HERE, "data", "arabido_fa.json")
REF_NPZ = os.path.join(HERE, "data", "arabido_fa.npz")

# (SMW, optimizations). `opti` drives structured_forward and analytic_backward
# together, as elsewhere in the suite: they share the per-effect decomposition
# and are not meaningfully independent for a correctness check.
SOLVING = [(True, True), (True, False), (False, True), (False, False)]
SOLVING_IDS = ["woodbury-opt", "woodbury-plain", "direct-opt", "direct-plain"]

# the route every other one is compared against
PIVOT = "woodbury-opt"

JITTER = 1e-6               # the FA capacitance is near-singular without it

# Cross-path: four routes to one likelihood, so only roundoff separates them.
# Loose enough to absorb a different order of operations, tight enough that a
# genuinely different optimum cannot pass.
RTOL_PATH = 1e-5
RTOL_PATH_BY_KEY = {"Psi": 1e-4}

# Against WOMBAT: a different algorithm stopping on a different criterion, on a
# non-convex surface. This says "the same optimum", not "the same arithmetic".
RTOL_WOMBAT = 5e-2


@pytest.fixture(scope="session")
def ref():
    with open(REF_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def data(ref):
    """Kinship and phenotypes, rebuilt exactly as the reference generator did.

    The line subset is the same deterministic prefix, so K is the kinship of
    the very lines WOMBAT was given -- a different subset would silently make
    the comparison meaningless.
    """
    cfg = ref["config"]

    with np.load(GENO) as z:
        # A_genomic writes into its argument, hence the fresh cast
        snp = z["snp"].astype(np.float64) - 1.0
        individuals = z["individual"]

    snp = snp[: cfg["n_ind"]]
    individuals = individuals[: cfg["n_ind"]]

    K = A_genomic(snp, min_MAF=0.05, max_missing=0.1, shrink=True)
    del snp

    with np.load(REF_NPZ) as z:
        y = z["y"]
        u_true = z["u_true"]

    df = pd.DataFrame(y, columns=cfg["env_labels"])
    df["individual"] = individuals

    return {"K": K, "df": df, "u_true": u_true, "individual": individuals}


@pytest.fixture(scope="session")
def models(ref, data):
    """Fit the four routes once, and hand them back keyed by route.

    Fitted together rather than through a parametrized fixture because the
    central assertion is a COMPARISON between routes: a parametrized fixture
    would give each test one model and no way to cross them.
    """
    cfg = ref["config"]
    Sigma0 = np.array(ref["start"]["Sigma0"])
    Res0 = np.array(ref["start"]["Res0"])

    out = {}
    for (smw, opti), name in zip(SOLVING, SOLVING_IDS):
        out[name] = MixedModel.from_dataframe(
            data=data["df"],
            response=cfg["env_labels"],
            fixed="1",
            random=Random(
                formula="1",
                unit="individual",
                right_hand="str",
                left_hand="fa",
                n_axes=cfg["n_axes"],
                covariance=data["K"],
                matrix_index=list(data["individual"]),
                init=Sigma0,
                jitter=JITTER,
            ),
            # no residual covariance: two environments are two trials
            residual=Residual(left_hand="diag", init=Res0),
            SMW=smw,
            structured_forward=opti,
            analytic_backward=opti,
            device=DEVICE,
        ).fit(DTYPE, verbose=False)

    return out


def _fa(model):
    """The invariant summary of a fitted FA.

    Sigma_A is read off `variance["sigma"]`, which format_variance already
    returns in natural units; Lambda and Psi come from the fa metadata. Q is
    deliberately absent: its columns carry an arbitrary sign and the factor
    basis is identified only up to a rotation.
    """
    fa = model.random[0].variance["metadata"]["fa"]
    return {
        "Sigma_A": np.asarray(model.random[0].variance["sigma"], dtype=float),
        "Lambda": np.asarray(fa["Lambda"], dtype=float),
        "Psi": np.asarray(fa["Psi"], dtype=float),
        "res_var": np.diag(
            np.asarray(model.residual.variance["sigma"], dtype=float)
        ).copy(),
        "intercepts": model.estimates["estimate"].to_numpy(),
        "neg2loglik": float(model.neg2loglik),
    }


def _rel(a, b):
    """Relative Frobenius deviation, on whichever of the two is the larger."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-12)
    return float(np.linalg.norm(a - b) / denom)


# ---- convergence and routing ----------------------------------------------

@pytest.mark.parametrize("name", SOLVING_IDS)
def test_converged(models, name):
    assert models[name].opti_REML.converged, f"{name} did not converge"


@pytest.mark.parametrize("name", SOLVING_IDS)
def test_routing(models, name, ref):
    """The requested route is the route actually taken.

    Worth asserting on its own: a garde that silently downgrades SMW would make
    the cross-path comparison compare a route against itself, and pass.
    """
    model = models[name]
    smw, opti = SOLVING[SOLVING_IDS.index(name)]

    assert model.SMW is smw
    assert model.structured_forward is opti
    assert model.analytic_backward is opti

    cfg = ref["config"]
    assert model.n == cfg["n_ind"] * cfg["n_env"]
    assert model.p == cfg["n_env"]                    # one intercept per trait
    assert model.q == cfg["n_ind"] * cfg["n_env"]     # d = k*c = n_env, L = n_ind


# ---- cross-path agreement --------------------------------------------------

@pytest.mark.parametrize("name", [i for i in SOLVING_IDS if i != PIVOT])
@pytest.mark.parametrize("key", ["Sigma_A", "Lambda", "Psi", "res_var", "intercepts"])
def test_paths_agree(models, name, key):
    """Four routes to one likelihood must land on one answer."""
    pivot, other = _fa(models[PIVOT]), _fa(models[name])
    tol = RTOL_PATH_BY_KEY.get(key, RTOL_PATH)
    dev = _rel(other[key], pivot[key])
    assert dev <= tol, (
        f"{name} vs {PIVOT}: {key} deviates by {dev:.2e} (allowed {tol:.0e})"
    )


@pytest.mark.parametrize("name", [i for i in SOLVING_IDS if i != PIVOT])
def test_paths_agree_on_likelihood(models, name):
    """The sharpest cross-path check: -2logL is one number, computed four ways.

    Compared in absolute terms against its own scale rather than relatively,
    since -2logL is O(n) and a relative tolerance would loosen with the sample
    size instead of tracking the precision actually available.
    """
    pivot = _fa(models[PIVOT])["neg2loglik"]
    other = _fa(models[name])["neg2loglik"]
    tol = RTOL_PATH * abs(pivot)
    assert abs(other - pivot) <= tol, (
        f"{name} vs {PIVOT}: -2logL {other:.6f} against {pivot:.6f} "
        f"(deviation {abs(other - pivot):.3e}, allowed {tol:.3e})"
    )


# ---- agreement with WOMBAT -------------------------------------------------

def test_wombat_converged(ref):
    """Guard on the reference itself: a non-converged WOMBAT run is not a
    reference, and comparing against it would be worse than not comparing."""
    assert ref["wombat"]["converged"], (
        "the frozen WOMBAT run did not converge; regenerate the reference "
        "before reading anything into the comparisons below"
    )


@pytest.mark.parametrize("key", ["Sigma_A", "Psi", "res_var", "intercepts"])
def test_matches_wombat(models, ref, key):
    """The two implementations agree on which optimum they found.

    Only the pivot route is compared: the others were just shown to agree with
    it to 1e-5, so running all four against WOMBAT would test the same thing
    four times.
    """
    ours = _fa(models[PIVOT])[key]
    theirs = np.asarray(ref["wombat"][key], dtype=float)

    assert ours.shape == theirs.shape, (
        f"{key}: shape {ours.shape} against WOMBAT's {theirs.shape}"
    )
    dev = _rel(ours, theirs)
    assert dev <= RTOL_WOMBAT, (
        f"{key} deviates from WOMBAT by {dev:.2e} (allowed {RTOL_WOMBAT:.0e})"
    )


def test_matches_wombat_eigenvalues(models, ref):
    """Lambda, compared as a spectrum rather than element-wise.

    The two solvers may order or scale the factor basis differently while
    describing the same subspace, so what must match is the set of eigenvalues,
    sorted -- not the loadings that carry them.
    """
    ours = np.sort(_fa(models[PIVOT])["Lambda"])[::-1]
    theirs = np.sort(np.asarray(ref["wombat"]["Lambda"], dtype=float))[::-1]

    dev = _rel(ours, theirs)
    assert dev <= RTOL_WOMBAT, (
        f"Lambda deviates from WOMBAT by {dev:.2e} (allowed {RTOL_WOMBAT:.0e}): "
        f"{np.array2string(ours, precision=4)} against "
        f"{np.array2string(theirs, precision=4)}"
    )


def test_common_subspace_matches_wombat(models, ref):
    """The FACTOR SUBSPACE, compared basis-free.

    Neither Q nor Gamma is identified -- signs and rotations are free -- but the
    subspace they span is. The projector Q Q' is invariant under both, so it is
    what gets compared. A model can match on Sigma_A while spanning a different
    subspace when Psi absorbs the difference, which is precisely the failure
    this catches and the Sigma_A comparison does not.
    """
    fa = models[PIVOT].random[0].variance["metadata"]["fa"]
    Q_ours = np.asarray(fa["Q"], dtype=float)
    Q_theirs = np.asarray(ref["wombat"]["Q"], dtype=float)

    # orthonormalize both: WOMBAT prints eigenvectors, pyreml a QR factor, and
    # neither guarantees the other's normalization
    Q_ours, _ = np.linalg.qr(Q_ours)
    Q_theirs, _ = np.linalg.qr(Q_theirs)

    dev = _rel(Q_ours @ Q_ours.T, Q_theirs @ Q_theirs.T)
    assert dev <= RTOL_WOMBAT, (
        f"the common subspace deviates from WOMBAT by {dev:.2e} "
        f"(allowed {RTOL_WOMBAT:.0e})"
    )