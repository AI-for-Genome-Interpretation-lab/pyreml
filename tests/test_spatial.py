import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

from pyreml import MixedModel, Random, larix as DF

DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "spatial.json") as f:
    EXPECTED = json.load(f)

with open(DATA_DIR / "spatial_pred.json") as f:
    EXPECTED_PRED = json.load(f)


def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

@pytest.fixture()
def df():
    df = DF.copy()
    df = df[df["year"] == 2000]
    df["ID"] = np.arange(len(df))
    return df

@pytest.fixture
def D_train(df):
    return _pairwise_dist(df[["X", "Y"]].to_numpy())

@pytest.fixture
def pred_inputs(df):
    df_pred = df[
        (df["BLOC"].isin([f"B{i}" for i in range(13, 17)]))
    ].copy()

    n = len(df)
    m = len(df_pred)

    coords_train = df[["X", "Y"]].to_numpy()
    coords_pred  = df_pred[["X", "Y"]].to_numpy()
    coords_full  = np.vstack([coords_train, coords_pred])

    D = _pairwise_dist(coords_full)
    K = np.exp(-EXPECTED["rho"] * D)
    return {
        "D_full": D,
        "K_full": K,
        "matrix_index": list(range(n + m)),
        "n": n,
        "m": m,
    }

@pytest.fixture(params=[True, False], ids=["woodbury", "direct"])
def mod_ar(df, D_train, request):
    return MixedModel.from_dataframe(
        data=df,
        response="height",
        fixed="1",
        random=Random(
            unit="ID",
            right_hand="ar",
            distance=D_train,
            matrix_index=df["ID"].tolist(),
        ),
        SMW=request.param,
    ).fit()


@pytest.fixture(params=[True, False], ids=["woodbury", "direct"])
def mod_str(df, D_train, request):
    K = np.exp(-EXPECTED["rho"] * D_train)
    return MixedModel.from_dataframe(
        data=df,
        response="height",
        fixed="1",
        random=Random(
            unit="ID",
            right_hand="str",
            covariance=K,
            matrix_index=df["ID"].tolist(),
        ),
        SMW=request.param,
    ).fit()


def _Vu(mod):
    return float(torch.exp(mod.random[0].log_S).detach().numpy())


def _Ve(mod):
    return float(torch.exp(mod.residual.log_S).detach().numpy())


def _blup(mod):
    # ar/str: k = c = 1, so uhat order is the level order = self.index = ID 0..n-1.
    return mod.random[0].uhat.detach().numpy().ravel()


class TestSpatialAR:

    def test_convergence(self, mod_ar):
        assert mod_ar.opti_REML.converged is True

    def test_rho(self, mod_ar):
        actual = mod_ar.random[0].variance["metadata"]["ar"]
        np.testing.assert_allclose(actual, EXPECTED["rho"], rtol=1.2e-4) # woodbury

    def test_Vu(self, mod_ar):
        np.testing.assert_allclose(_Vu(mod_ar), EXPECTED["Vu"], rtol=1.1e-4) # woodbury

    def test_Ve(self, mod_ar):
        np.testing.assert_allclose(_Ve(mod_ar), EXPECTED["Ve"], rtol=1e-4)

    def test_intercept(self, mod_ar):
        actual = mod_ar.estimates["estimate"].to_numpy()
        np.testing.assert_allclose(actual, [EXPECTED["beta"]], rtol=1e-4, atol=1e-6)

    def test_eev_intercept(self, mod_ar):
        np.testing.assert_allclose(
            mod_ar.EEV.item(), EXPECTED["eev_intercept"], rtol=2e-4
        )

    def test_blup(self, mod_ar):
        np.testing.assert_allclose(
            _blup(mod_ar), EXPECTED["blup"], atol= 1e-3
        )

    def test_pev_diag(self, mod_ar):
        pev = mod_ar.random[0].PEV.detach().numpy()
        np.testing.assert_allclose(
            np.diag(pev), EXPECTED["pev_diag"], rtol=1e-3, atol=1e-6
        )

    def test_predict(self, mod_ar, pred_inputs):
        out = mod_ar.random[0].predict(
            matrix_index=pred_inputs["matrix_index"],
            distance=pred_inputs["D_full"],
        )
        actual = out["prediction"].to_numpy()
        np.testing.assert_allclose(
            actual, EXPECTED_PRED["blup_pred"], atol= 1e-3
        )


class TestSpatialSTR:

    def test_convergence(self, mod_str):
        assert mod_str.opti_REML.converged is True

    def test_Vu(self, mod_str):
        np.testing.assert_allclose(_Vu(mod_str), EXPECTED["Vu"], rtol=1e-4)

    def test_Ve(self, mod_str):
        np.testing.assert_allclose(_Ve(mod_str), EXPECTED["Ve"], rtol=1e-4)

    def test_intercept(self, mod_str):
        actual = mod_str.estimates["estimate"].to_numpy()
        np.testing.assert_allclose(actual, [EXPECTED["beta"]], rtol=1e-4, atol=1e-6)

    def test_eev_intercept(self, mod_str):
        np.testing.assert_allclose(
            mod_str.EEV.item(), EXPECTED["eev_intercept"], rtol=1e-4
        )

    def test_blup(self, mod_str):
        np.testing.assert_allclose(
            _blup(mod_str), EXPECTED["blup"], atol= 5e-4 # !
        )

    def test_pev_diag(self, mod_str):
        pev = mod_str.random[0].PEV.detach().numpy()
        np.testing.assert_allclose(
            np.diag(pev), EXPECTED["pev_diag"], rtol = 1e-5
        )

    def test_predict(self, mod_str, pred_inputs):
        out = mod_str.random[0].predict(
            matrix_index=pred_inputs["matrix_index"],
            covariance=pred_inputs["K_full"],
        )
        actual = out["prediction"].to_numpy()
        np.testing.assert_allclose(
            actual, EXPECTED_PRED["blup_pred"], atol= 5e-4 # !
        )
