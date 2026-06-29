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

    spat_dist   <-  dist, str, eucl(2D)              Euclidean 2D kernel
    spat_iso  <-  ar_iso(2D)                        Manhattan (diamond) kernel
    spat_ani  <-  ar_ani(2D)                        anisotropic kernel

The 1D variants (eucl/ar_iso/ar_ani on a single axis) are intentionally NOT
tested. In 1D the spatial random effect collapses onto a random global intercept
(K -> J as rho -> 0), confounded with the fixed intercept; the REML optimum is a
boundary degeneracy at rho -> 0 / Vu -> inf rather than an interior point. The
campaign's premise is a well-defined interior optimum to match against rrBLUP,
which 1D violates. See the demonstration cell in the reference generator
(`make_spatial_refs.py`) for the reproduction of that degeneracy.

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
(kriging hold-out). All cases run with the direct solver; every case except the
two 2D AR structures also runs with Woodbury (completing the full 2D grid makes
the Woodbury latent system prohibitively large).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from pyreml import MixedModel, Random, larix as DF


DATA_DIR = Path(__file__).parent / "data"


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
    """One explicitly permitted case/solver combination."""

    case: dict
    smw: bool

    @property
    def id(self) -> str:
        solver = "woodbury" if self.smw else "direct"
        return f"{solver}-{self.case['id']}"


# All cases run directly. All but the two 2D AR structures also run with Woodbury.
RUNS = [
    Run(case=case, smw=True)
    for case in CASES
    if case["id"] not in {"ar_iso", "ar_ani"}
] + [
    Run(case=case, smw=False)
    for case in CASES
]


# --------------------------------------------------------------------------- #
# Helpers / data
# --------------------------------------------------------------------------- #
def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _full(df):
    """Full year-2000 grid; one level per row, unique (X, Y)."""
    df = df[df["year"] == 2000].copy()
    df["ID"] = np.arange(len(df))
    return df


def _dataset(case):
    return _full(DF)


def _holdout(case, data):
    """Kriging hold-out re-predicted as new levels (blocks B13..B16).

    NOTE these are observed rows whose coordinates also appear in the training
    set: with integer labels (dist/str) they are distinct new levels, but with
    coordinate-tuple labels (eucl/ar) the prediction generator offsets them off
    the grid so they become genuine new levels. See _holdout_coords / PRED_OFFSET
    in the reference generator.
    """
    return data[data["BLOC"].isin([f"B{i}" for i in range(13, 17)])]


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

    For dist / str the train+pred levels are labelled by integers (range(n+m))
    and the kernel is supplied as an (n+m, n+m) distance / covariance.

    For the coordinate kernels there is no separate label: matrix_index IS the
    list of coordinate tuples, train then new. Train tuples are reconstructed
    from `data` in row order, which on this dataset coincides with the model's
    first-occurrence level order; predict realigns positionally anyway.
    """
    data = _dataset(case)
    df_pred = _holdout(case, data)

    n, m = len(data), len(df_pred)

    coords_train = data[["X", "Y"]].to_numpy()
    coords_pred = df_pred[["X", "Y"]].to_numpy()
    coords_full = np.vstack([coords_train, coords_pred])

    out = {
        "matrix_index": list(range(n + m)),
        "coords_full": coords_full,              # (n+m, 2): X then Y
        "D_full": _pairwise_dist(coords_full),   # Euclidean, for dist/str
        "n": n,
        "m": m,
    }

    if case["coords"] is not None:
        ncol = len(case["coords"])
        sel = coords_full[:, :ncol]
        out["tuples_full"] = [tuple(row) for row in sel]

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
    ).fit()


def _Vu(mod):
    return float(torch.exp(mod.random[0].log_S).detach().numpy())


def _Ve(mod):
    return float(torch.exp(mod.residual.log_S).detach().numpy())


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
    u = mod.random[0].uhat.detach().numpy().ravel()
    return _observed(mod, u)


# --------------------------------------------------------------------------- #
# Assertions — one class, parametrized over the explicitly permitted RUNS.
# --------------------------------------------------------------------------- #
class TestSpatial:

    def test_convergence(self, mod):
        assert mod.opti_REML.converged is True

    def test_rho(self, mod, case, expected):
        meta = mod.random[0].variance["metadata"]

        if not case["has_rate"]:
            assert "rho" not in meta
            return

        actual = meta["rho"]

        if case["rho_is_list"]:
            assert isinstance(actual, list)
            assert len(actual) == case["n_rho"]
            np.testing.assert_allclose(actual, expected["rho"], atol=1e-4)
        else:
            assert not isinstance(actual, list)
            np.testing.assert_allclose(actual, expected["rho"][0], atol=1e-4)

    def test_Vu(self, mod, expected):
        actual = _Vu(mod)
        desired = expected["Vu"]

        if not (desired > 1e6 and actual > 1e6):
            np.testing.assert_allclose(actual, desired, rtol=1e-4)


    def test_Ve(self, mod, expected):
        actual = _Ve(mod)
        desired = expected["Ve"]

        if not (desired > 1e6 and actual > 1e6):
            np.testing.assert_allclose(actual, desired, rtol=1e-4)

    def test_intercept(self, mod, expected):
        actual = mod.estimates["estimate"].to_numpy()
        np.testing.assert_allclose(actual, [expected["beta"]], rtol=1e-4, atol=1e-6)

    def test_eev_intercept(self, mod, expected):
        np.testing.assert_allclose(mod.EEV.item(), expected["eev_intercept"], rtol=2e-4)

    def test_blup(self, mod, expected):
        np.testing.assert_allclose(_blup(mod), expected["blup"], atol=1e-3)

    def test_pev_diag(self, mod, expected):
        pev = mod.random[0].PEV.detach().numpy()
        diag = _observed(mod, np.diag(pev))
        np.testing.assert_allclose(diag, expected["pev_diag"], rtol=1e-3, atol=1e-6)

    def test_predict(self, mod, case, pred_inputs, expected, expected_pred):
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
        np.testing.assert_allclose(actual, expected_pred["blup_pred"], atol=1e-3)