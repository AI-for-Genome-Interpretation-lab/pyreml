"""
Spatial / decay-kernel variance structures — target specification.

This campaign exercises the `right_hand` decay family on the year-2000 larix
trial against rrBLUP references:

    dist     supplied Euclidean distance matrix      exp(-rho * D)
    str      known covariance (rho fixed)            exp(-rho* * D)
    eucl     internal Euclidean distance, n-D        exp(-rho * ||dx||_2)
    ar_iso   separable AR, one shared rate           exp(-rho * sum|dx_a|)   (L1)
    ar_ani   separable AR, one rate per axis         exp(-sum rho_a |dx_a|)

Reference sharing:

    spat_dist   <-  dist, str, eucl                  Euclidean kernel
    spat_iso    <-  ar_iso                            Manhattan (diamond) kernel
    spat_ani    <-  ar_ani                            anisotropic kernel

Runs on a contrasted-but-close bloc subset (SUBSET_BLOCS) chosen so the AR
kernels sit in an identifiable regime (rho * spacing moderate, neither flat nor
diagonal). The 1D variants are intentionally NOT tested: in 1D the spatial
random effect collapses onto a random global intercept (K -> J as rho -> 0),
confounded with the fixed intercept, so the REML optimum is a boundary
degeneracy rather than an interior point. See the demonstration cell in
`make_spatial_refs.py`.

Identity of a level. With the coordinate kernels (eucl, ar_iso, ar_ani) `unit`
is no longer a single grouping column but the LIST of coordinate columns: a
level is a distinct coordinate tuple, enumerated in first-occurrence order. The
dist / str cases keep the categorical `unit="ID"` plus an explicit
distance / covariance ordered by `matrix_index`.

Grid vs observed levels. ar_iso / ar_ani complete the full integer grid spanned
by the coordinates: the model's internal uhat / PEV / index then run over ALL
grid cells (empty cells included), while the rrBLUP references run over the
OBSERVED levels only. `level_cell` (set by make_coords) maps each observed level
to its grid cell, in the same first-occurrence order as the reference; the BLUP
and PEV assertions collapse the grid down to the observed levels through it.
eucl carries no grid, so level_cell is absent and the collapse is a no-op.

rho reporting. The fitted rate(s) live in metadata["rho"]. It is a plain scalar
for the single-rate kernels (dist, eucl, ar_iso) and a list for ar_ani (one rate
per axis). str estimates no rate and exposes no "rho" key.

Each reference is a pair `spat_{ref}.json` (fit) + `spat_{ref}_pred.json`
(kriging hold-out). All cases run with the direct solver; both AR structures
also run only directly (completing the full grid makes the Woodbury latent
system prohibitively large).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from pyreml import MixedModel, Random, larix as DF


DATA_DIR = Path(__file__).parent / "data"

# MUST be kept identical to SUBSET_BLOCS / PRED_OFFSET / _grid / _holdout in
# make_spatial_refs.py: the model and the reference must see the exact same rows
# in the exact same order.
SUBSET_BLOCS = ["B3", "B13"]
PRED_OFFSET = 0.5


# --------------------------------------------------------------------------- #
# Dtypes and tolerances.
# --------------------------------------------------------------------------- #
DTYPES = [torch.float64, torch.float32]


RHO_ATOL = 1e-3
VAR_RTOL = 3e-3
BETA_RTOL = 1e-3
EEV_RTOL = 3e-4
BLUP_RTOL = 3e-3
PEV_RTOL = 3e-3
PRED_ATOL = 2e-2


# --------------------------------------------------------------------------- #
# Cases.
#
#   token        -> right_hand passed to Random
#   coords       -> coordinate columns for the kernel-based cases; these columns
#                   ARE the `unit` for eucl / ar_iso / ar_ani. None for dist/str
#                   (which keep unit="ID").
#   ref          -> reference basename: spat_{ref}.json / spat_{ref}_pred.json.
#   has_rate     -> whether a decay rate is estimated (str: no).
#   n_rho        -> number of estimated rates (length of the rate vector).
#   rho_is_list  -> metadata["rho"] is a list (ar_ani) vs a scalar (others).
#   gridded      -> the kernel completes an integer grid (ar_iso / ar_ani):
#                   internal uhat/PEV span the grid and must be collapsed.
#   pred         -> predict() input channel: distance | covariance | coords
#                   ("coords" means the tuples travel inside matrix_index).
# --------------------------------------------------------------------------- #
CASES = [
    dict(
        id="dist", token="dist", coords=None, ref="dist",
        has_rate=True, n_rho=1, rho_is_list=False, gridded=False,
        pred="distance",
    ),
    dict(
        id="str", token="str", coords=None, ref="dist",
        has_rate=False, n_rho=0, rho_is_list=False, gridded=False,
        pred="covariance",
    ),
    dict(
        id="eucl", token="eucl", coords=["X", "Y"], ref="dist",
        has_rate=True, n_rho=1, rho_is_list=False, gridded=False,
        pred="coords",
    ),
    dict(
        id="ar_iso", token="ar_iso", coords=["X", "Y"], ref="iso",
        has_rate=True, n_rho=1, rho_is_list=False, gridded=True,
        pred="coords",
    ),
    dict(
        id="ar_ani", token="ar_ani", coords=["X", "Y"], ref="ani",
        has_rate=True, n_rho=2, rho_is_list=True, gridded=True,
        pred="coords",
    ),
]


@dataclass(frozen=True)
class Run:
    """One explicitly permitted case / solver / dtype combination."""

    case: dict
    smw: bool
    dtype: torch.dtype

    @property
    def dtype_id(self) -> str:
        if self.dtype == torch.float64:
            return "double"
        if self.dtype == torch.float32:
            return "float"
        return str(self.dtype).replace("torch.", "")

    @property
    def solver_id(self) -> str:
        return "woodbury" if self.smw else "direct"

    @property
    def id(self) -> str:
        return f"{self.dtype_id}-{self.solver_id}-{self.case['id']}"


# Preserve the original permitted solver matrix, then duplicate by dtype:
# - all cases run directly;
# - only non-AR structures run with Woodbury.
BASE_RUNS = [
    (case, True)
    for case in CASES
    if case["id"] not in {"ar_iso", "ar_ani"}
] + [
    (case, False)
    for case in CASES
]

RUNS = [
    Run(case=case, smw=smw, dtype=dtype)
    for dtype in DTYPES
    for case, smw in BASE_RUNS
]


# --------------------------------------------------------------------------- #
# Helpers / data
# --------------------------------------------------------------------------- #
def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _grid(df):
    """Year-2000 grid restricted to SUBSET_BLOCS; one level per row, unique (X, Y)."""
    df = df.loc[(df["year"] == 2000) & df["BLOC"].isin(SUBSET_BLOCS)].copy()
    df["ID"] = np.arange(len(df))
    return df


def _holdout(data):
    """Kriging hold-out: every 3rd row, re-predicted as NEW levels via PRED_OFFSET.
    Deterministic and independent of the bloc choice.
    """
    return data.iloc[::3]


def _dataset(case):
    return _grid(DF)


def _is_float(run):
    return run.dtype == torch.float32


def _is_double(run):
    return run.dtype == torch.float64


def _rtol(run, double_value, float_value):
    return double_value if _is_double(run) else float_value


def _atol(run, double_value, float_value):
    return double_value if _is_double(run) else float_value


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(params=RUNS, ids=lambda run: run.id)
def run(request):
    return request.param


@pytest.fixture
def case(run):
    return run.case


@pytest.fixture
def expected(case):
    with open(DATA_DIR / f"spat_{case['ref']}.json") as f:
        return json.load(f)


@pytest.fixture
def expected_pred(case):
    with open(DATA_DIR / f"spat_{case['ref']}_pred.json") as f:
        return json.load(f)


@pytest.fixture
def pred_inputs(case):
    """Augmented train+pred geometry for the kriging hold-out.

    Held-out rows are re-predicted as GENUINELY NEW levels by shifting their
    coordinates by PRED_OFFSET, exactly as the reference generator. Without the
    shift a held-out coordinate duplicates a training one: the coordinate kernels
    then return zero predictions, and the integer-labelled kernels (dist/str)
    condition a point on its own position. The shift stays float through predict
    (the coordinate prediction path is dense, no grid rounding).

    dist / str: integer level labels (range(n+m)), kernel supplied as an
    (n+m, n+m) distance / covariance on the shifted coordinates.
    coordinate kernels: matrix_index IS the tuple list, train (unshifted) then
    new (shifted).
    """
    data = _grid(DF)
    df_pred = _holdout(data)

    n, m = len(data), len(df_pred)

    coords_train = data[["X", "Y"]].to_numpy(dtype=float)
    coords_pred = df_pred[["X", "Y"]].to_numpy(dtype=float) + PRED_OFFSET
    coords_full = np.vstack([coords_train, coords_pred])

    out = {
        "matrix_index": list(range(n + m)),
        "coords_full": coords_full,
        "D_full": _pairwise_dist(coords_full),
        "n": n,
        "m": m,
    }

    if case["coords"] is not None:
        ncol = len(case["coords"])
        out["tuples_full"] = [tuple(row) for row in coords_full[:, :ncol]]

    return out


@pytest.fixture
def mod(run, expected):
    case = run.case
    data = _dataset(case)
    token = case["token"]

    if token == "dist":
        D = _pairwise_dist(data[["X", "Y"]].to_numpy())
        eff = Random(
            unit="ID",
            right_hand="dist",
            distance=D,
            matrix_index=data["ID"].tolist(),
        )

    elif token == "str":
        D = _pairwise_dist(data[["X", "Y"]].to_numpy())
        K = np.exp(-expected["rho"][0] * D)
        eff = Random(
            unit="ID",
            right_hand="str",
            covariance=K,
            matrix_index=data["ID"].tolist(),
        )

    else:
        # eucl / ar_iso / ar_ani: the coordinate columns ARE the unit.
        eff = Random(
            unit=case["coords"],
            right_hand=token,
        )

    return MixedModel.from_dataframe(
        data=data,
        response="height",
        fixed="1",
        random=eff,
        SMW=run.smw,
        dtype=run.dtype,
    ).fit()


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #
def _Vu(mod):
    return float(torch.exp(mod.random[0].log_S).detach().cpu().numpy())


def _Ve(mod):
    return float(torch.exp(mod.residual.log_S).detach().cpu().numpy())


def _observed(mod, vec):
    """Collapse a grid-ordered vector down to the observed levels.

    For the gridded kernels (ar_iso / ar_ani) uhat / diag(PEV) span every grid
    cell; level_cell maps each observed level (first-occurrence order, matching
    the reference) to its grid cell. eucl / dist / str carry no grid, so
    level_cell is absent and the vector is returned untouched.
    """
    cell = getattr(mod.random[0], "level_cell", None)
    return vec if cell is None else vec[np.asarray(cell)]


def _blup(mod):
    # Decay kernels: k = c = 1, so uhat is ordered by level only.
    u = mod.random[0].uhat.detach().cpu().numpy().ravel()
    return _observed(mod, u)


# --------------------------------------------------------------------------- #
# Known non-threshold issues kept explicit.
# --------------------------------------------------------------------------- #
def _xfail_ar_ani_float32(run):
    if run.case["id"] == "ar_ani" and _is_float(run):
        pytest.xfail("ar_ani + float32 is unstable on SUBSET_BLOCS; keep as known model issue.")


def _xfail_ar_ani_known_Ve(run):
    if run.case["id"] == "ar_ani":
        pytest.xfail("ar_ani degenerate on SUBSET_BLOCS (Ve -> 0 plateau)")


def _xfail_str_float32_optimizer_issue(run):
    if run.case["id"] == "str" and _is_float(run):
        pytest.xfail(
            "str + float32 currently converges to a bad variance basin "
            "(Vu ~ 6, Ve ~ 4817 instead of Vu ~ 3212, Ve ~ 1591); "
            "do not hide this behind loose tolerances."
        )


def _xfail_known_float32_issue(run):
    _xfail_ar_ani_float32(run)
    _xfail_str_float32_optimizer_issue(run)


# --------------------------------------------------------------------------- #
# Assertions — parametrized over original permitted runs x dtype.
# --------------------------------------------------------------------------- #
class TestSpatial:

    def test_convergence(self, mod):
        assert mod.opti_REML.converged is True

    def test_rho(self, run, mod, case, expected):
        _xfail_ar_ani_float32(run)

        meta = mod.random[0].variance["metadata"]

        if not case["has_rate"]:
            assert "rho" not in meta
            return

        actual = meta["rho"]
        atol = _atol(run, RHO_ATOL, RHO_ATOL)

        if case["rho_is_list"]:
            assert isinstance(actual, list)
            assert len(actual) == case["n_rho"]
            np.testing.assert_allclose(actual, expected["rho"], atol=atol)
        else:
            assert not isinstance(actual, list)
            np.testing.assert_allclose(actual, expected["rho"][0], atol=atol)

    def test_Vu(self, run, mod, expected):
        _xfail_known_float32_issue(run)
        actual = _Vu(mod)
        desired = expected["Vu"]
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=_rtol(run, VAR_RTOL, VAR_RTOL),
        )

    def test_Ve(self, run, mod, expected):
        # This is the original documented ar_ani degeneracy. Keep it for both
        # dtypes, but do not spread it to every double assertion.
        _xfail_ar_ani_known_Ve(run)
        _xfail_str_float32_optimizer_issue(run)

        actual = _Ve(mod)
        desired = expected["Ve"]
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=_rtol(run, VAR_RTOL, VAR_RTOL),
        )

    def test_intercept(self, run, mod, expected):
        _xfail_ar_ani_float32(run)
        actual = mod.estimates["estimate"].to_numpy()[0]
        desired = expected["beta"]
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=_rtol(run, BETA_RTOL, BETA_RTOL),
        )

    def test_eev_intercept(self, run, mod, expected):
        _xfail_known_float32_issue(run)
        np.testing.assert_allclose(
            mod.EEV.item(),
            expected["eev_intercept"],
            rtol=_rtol(run, EEV_RTOL, EEV_RTOL),
        )

    def test_blup(self, run, mod, expected):
        _xfail_known_float32_issue(run)
        np.testing.assert_allclose(
            _blup(mod),
            expected["blup"],
            rtol=_rtol(run, BLUP_RTOL, BLUP_RTOL),
        )

    def test_pev_diag(self, run, mod, expected):
        _xfail_known_float32_issue(run)
        pev = mod.random[0].PEV.detach().cpu().numpy()
        diag = _observed(mod, np.diag(pev))
        np.testing.assert_allclose(
            diag,
            expected["pev_diag"],
            rtol=_rtol(run, PEV_RTOL, PEV_RTOL),
        )

    def test_predict(self, run, mod, case, pred_inputs, expected, expected_pred):
        _xfail_known_float32_issue(run)

        kind = case["pred"]

        if kind == "distance":
            out = mod.random[0].predict(
                matrix_index=pred_inputs["matrix_index"],
                distance=pred_inputs["D_full"],
            )

        elif kind == "covariance":
            K_full = np.exp(-expected["rho"][0] * pred_inputs["D_full"])
            out = mod.random[0].predict(
                matrix_index=pred_inputs["matrix_index"],
                covariance=K_full,
            )

        else:
            # coordinate kernels: the tuples ARE the level labels and travel
            # inside matrix_index; predict rebuilds the kernel from them.
            out = mod.random[0].predict(
                matrix_index=pred_inputs["tuples_full"],
            )

        actual = out["prediction"].to_numpy()
        np.testing.assert_allclose(
            actual,
            expected_pred["blup_pred"],
            atol=_atol(run, PRED_ATOL, PRED_ATOL),
        )