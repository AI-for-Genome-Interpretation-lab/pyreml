"""
Spatial / decay-kernel variance structures — target specification.

This campaign exercises the whole `right_hand` decay family on the year-2000
larix trial, against rrBLUP references:

    dist     supplied Euclidean distance matrix      exp(-rho * D)
    str      known covariance (rho fixed)            exp(-rho* * D)
    eucl     internal Euclidean distance, n-D        exp(-rho * ||dx||_2)
    ar_iso   separable AR, one shared rate           exp(-rho * sum|dx_a|)   (L1)
    ar_ani   separable AR, one rate per axis         exp(-sum rho_a |dx_a|)

Reference sharing (this is what makes the documented coincidences testable):

    spat_dist   <-  dist, str, eucl(2D)        Euclidean 2D kernel
    spat_1d     <-  eucl(1D), ar_iso(1D), ar_ani(1D)   1D identity
    spat_iso2d  <-  ar_iso(2D)                 Manhattan (diamond) kernel
    spat_ani2d  <-  ar_ani(2D)                 anisotropic kernel

Data. The 2D / dist / str cases run on the full year-2000 grid, where every
(X, Y) is unique. The coordinate kernels enforce one coordinate per level and no
duplicate coordinate across levels, so the 1D cases cannot run on the full grid
(X repeats down each column). They run instead on a TRANSECT: the most-populated
Y row, one observation per X. `unit` stays the categorical "ID"; only the data
narrows to a clean 1D series. The reference generator must build spat_1d on the
exact same transect.

Each reference is a pair `spat_{ref}.json` (fit) + `spat_{ref}_pred.json`
(kriging hold-out). `metadata["rho"]` is a list in every case (empty for `str`).

Nothing here runs until the structures are implemented and the JSON references
generated; the file is the contract those implementations must satisfy.
"""

import json
from pathlib import Path

import numpy as np
import torch
import pytest

from pyreml import MixedModel, Random, larix as DF

DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Cases. Each case is one variance structure scenario; it is crossed with the
# woodbury/direct SMW switch by the `mod` fixture.
#
#   token     -> right_hand passed to Random
#   coords    -> column list for coordinate-based kernels (None for dist/str)
#   ref       -> reference basename: loads spat_{ref}.json / spat_{ref}_pred.json
#                ref == "1d" also selects the transect dataset
#   has_rate  -> whether a decay rate is estimated (str: no)
#   n_rho     -> expected length of metadata["rho"]
#   pred      -> which keyword predict() receives: distance | covariance | coords
# --------------------------------------------------------------------------- #
CASES = [
    dict(id="dist",      token="dist",   coords=None,      ref="dist",   has_rate=True,  n_rho=1, pred="distance"),
    dict(id="str",       token="str",    coords=None,      ref="dist",   has_rate=False, n_rho=0, pred="covariance"),
    dict(id="eucl_2d",   token="eucl",   coords=["X", "Y"], ref="dist",   has_rate=True,  n_rho=1, pred="coords"),
    dict(id="eucl_1d",   token="eucl",   coords=["X"],      ref="1d",     has_rate=True,  n_rho=1, pred="coords"),
    dict(id="ar_iso_1d", token="ar_iso", coords=["X"],      ref="1d",     has_rate=True,  n_rho=1, pred="coords"),
    dict(id="ar_ani_1d", token="ar_ani", coords=["X"],      ref="1d",     has_rate=True,  n_rho=1, pred="coords"),
    dict(id="ar_iso_2d", token="ar_iso", coords=["X", "Y"], ref="iso2d",  has_rate=True,  n_rho=1, pred="coords"),
    dict(id="ar_ani_2d", token="ar_ani", coords=["X", "Y"], ref="ani2d",  has_rate=True,  n_rho=2, pred="coords"),
]


# --------------------------------------------------------------------------- #
# Helpers / data
# --------------------------------------------------------------------------- #
def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _full(df):
    """Full year-2000 grid; one level per row, unique (X, Y)."""
    df = df[df["year"] == 2000].copy()
    df["ID"] = np.arange(len(df))
    return df


def _transect(df):
    """1D series along X at the most-populated Y row, one observation per X.

    Deterministic and shared verbatim with the reference generator: the model
    and the reference must see the exact same rows in the exact same order.
    """
    df = _full(df)
    y0 = df["Y"].value_counts().idxmax()
    t = df[df["Y"] == y0].drop_duplicates(subset="X", keep="first").copy()
    t["ID"] = np.arange(len(t))
    return t


def _dataset(case):
    return _transect(DF) if case["ref"] == "1d" else _full(DF)


def _holdout(case, data):
    """Kriging hold-out re-predicted as new levels."""
    if case["ref"] == "1d":
        return data.iloc[::4]                          # every 4th transect point
    return data[data["BLOC"].isin([f"B{i}" for i in range(13, 17)])]


@pytest.fixture(params=CASES, ids=[c["id"] for c in CASES])
def case(request):
    return request.param


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
    """Augmented train+pred geometry for the kriging hold-out."""
    data = _dataset(case)
    df_pred = _holdout(case, data)

    n, m = len(data), len(df_pred)
    coords_train = data[["X", "Y"]].to_numpy()
    coords_pred = df_pred[["X", "Y"]].to_numpy()
    coords_full = np.vstack([coords_train, coords_pred])

    return {
        "matrix_index": list(range(n + m)),
        "coords_full": coords_full,            # (n+m, 2): X then Y
        "D_full": _pairwise_dist(coords_full),  # Euclidean, for dist/str
        "n": n,
        "m": m,
    }


@pytest.fixture(params=[True, False], ids=["woodbury", "direct"])
def mod(case, expected, request):
    data = _dataset(case)
    token = case["token"]

    if token == "dist":
        D = _pairwise_dist(data[["X", "Y"]].to_numpy())
        eff = Random(
            unit="ID", right_hand="dist",
            distance=D, matrix_index=data["ID"].tolist(),
        )
    elif token == "str":
        D = _pairwise_dist(data[["X", "Y"]].to_numpy())
        K = np.exp(-expected["rho"][0] * D)
        eff = Random(
            unit="ID", right_hand="str",
            covariance=K, matrix_index=data["ID"].tolist(),
        )
    else:  # eucl / ar_iso / ar_ani: coordinates read from the frame
        eff = Random(unit="ID", right_hand=token, coords=case["coords"])

    return MixedModel.from_dataframe(
        data=data,
        response="height",
        fixed="1",
        random=eff,
        SMW=request.param,
    ).fit()


def _Vu(mod):
    return float(torch.exp(mod.random[0].log_S).detach().numpy())


def _Ve(mod):
    return float(torch.exp(mod.residual.log_S).detach().numpy())


def _blup(mod):
    # decay kernels: k = c = 1, uhat order is the level order (self.index).
    return mod.random[0].uhat.detach().numpy().ravel()


# --------------------------------------------------------------------------- #
# Assertions — one class, parametrized over (case x SMW) via the `mod` fixture.
# Tolerances mirror the original AR/STR campaign; tighten per-case if needed.
# --------------------------------------------------------------------------- #
class TestSpatial:

    def test_convergence(self, mod):
        assert mod.opti_REML.converged is True

    def test_rho(self, mod, case, expected):
        if not case["has_rate"]:
            actual = mod.random[0].variance["metadata"]["rho"]
            assert list(actual) == []          # str: empty rate list
            return
        actual = mod.random[0].variance["metadata"]["rho"]
        assert len(actual) == case["n_rho"]    # rho is a list, coords order
        np.testing.assert_allclose(actual, expected["rho"], rtol=2e-4)

    def test_Vu(self, mod, expected):
        np.testing.assert_allclose(_Vu(mod), expected["Vu"], rtol=2e-4)

    def test_Ve(self, mod, expected):
        np.testing.assert_allclose(_Ve(mod), expected["Ve"], rtol=1e-4)

    def test_intercept(self, mod, expected):
        actual = mod.estimates["estimate"].to_numpy()
        np.testing.assert_allclose(actual, [expected["beta"]], rtol=1e-4, atol=1e-6)

    def test_eev_intercept(self, mod, expected):
        np.testing.assert_allclose(mod.EEV.item(), expected["eev_intercept"], rtol=2e-4)

    def test_blup(self, mod, expected):
        np.testing.assert_allclose(_blup(mod), expected["blup"], atol=1e-3)

    def test_pev_diag(self, mod, expected):
        pev = mod.random[0].PEV.detach().numpy()
        np.testing.assert_allclose(np.diag(pev), expected["pev_diag"], rtol=1e-3, atol=1e-6)

    def test_predict(self, mod, case, pred_inputs, expected, expected_pred):
        mi = pred_inputs["matrix_index"]
        kind = case["pred"]

        if kind == "distance":
            out = mod.random[0].predict(matrix_index=mi, distance=pred_inputs["D_full"])
        elif kind == "covariance":
            K_full = np.exp(-expected["rho"][0] * pred_inputs["D_full"])
            out = mod.random[0].predict(matrix_index=mi, covariance=K_full)
        else:  # coords
            ncol = len(case["coords"])
            coords = pred_inputs["coords_full"][:, :ncol]
            out = mod.random[0].predict(matrix_index=mi, coords=coords)

        actual = out["prediction"].to_numpy()
        np.testing.assert_allclose(actual, expected_pred["blup_pred"], atol=1e-3)