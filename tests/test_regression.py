import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pytest

from pyreml import MixedModel, Random, larix as DF

DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "regression_fixed.json") as f:
    EXPECTED_FIXED_REG = json.load(f)

with open(DATA_DIR / "regression_random.json") as f:
    EXPECTED_RANDOM_REG = json.load(f)


@pytest.fixture
def df():

    df = DF.copy()
    df = df[df["year"] == 2000]
    df["circumference"] = (df["circumference"] - df["circumference"].mean()) / df["circumference"].std()
    df["height"] = (df["height"] - df["height"].mean()) / df["height"].std()
    df["ID"] = np.arange(len(df))

    return df


@pytest.fixture
def mod_ols(df):
    return MixedModel.from_dataframe(
        data=df,
        response="height",
        fixed="1 + BLOC + circumference",
    ).fit()


@pytest.fixture(params=[True, False], ids=["woodbury", "direct"])
def mod_lmm(df, request):
    return MixedModel.from_dataframe(
        data=df,
        response="height",
        fixed="1 + circumference",
        random=Random(
            unit="BLOC",
            formula="1 + circumference",
            left_hand="full",
        ),
        SMW=request.param,
    ).fit()

class TestOLS:

    def test_do_REML(self, mod_ols):
        assert mod_ols.do_REML is False

    def test_random(self, mod_ols):
        assert mod_ols.random == []

    def test_response(self, mod_ols):
        assert mod_ols.response == ["height"]

    def test_beta(self, mod_ols):
        expected = EXPECTED_FIXED_REG["beta"]
        actual = mod_ols.estimates["estimate"].tolist()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)

    def test_eev(self, mod_ols):
        expected = EXPECTED_FIXED_REG["eev"]
        actual = mod_ols.EEV.detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_residuals(self, mod_ols):
        expected = EXPECTED_FIXED_REG["residuals"]
        actual = mod_ols.residual.table["residual"].tolist()
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_tvals(self, mod_ols):
        expected = EXPECTED_FIXED_REG["tvals"]
        actual = mod_ols.estimates["t"].tolist()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)

    def test_aic(self, mod_ols):
            expected = EXPECTED_FIXED_REG["aic"]
            actual = float(mod_ols.AIC)
            np.testing.assert_allclose(actual, expected, atol=1e-3)

class TestLMM:

    def test_convergence(self, mod_lmm):
        assert mod_lmm.opti_REML.converged is True

    def test_do_REML(self, mod_lmm):
        assert mod_lmm.do_REML is True

    def test_response(self, mod_lmm):
        assert mod_lmm.response == ["height"]

    def test_beta(self, mod_lmm):
        expected = EXPECTED_RANDOM_REG["beta"]
        actual = mod_lmm.estimates["estimate"].tolist()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_eev(self, mod_lmm):
        expected = EXPECTED_RANDOM_REG["eev"]
        actual = mod_lmm.EEV.detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_varcorr(self, mod_lmm):
        expected = EXPECTED_RANDOM_REG["varcorr"]
        actual = mod_lmm.random[0].build_S().detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_sigma_r(self, mod_lmm):
        expected = EXPECTED_RANDOM_REG["sigma_r"]
        actual = float(torch.exp(mod_lmm.residual.log_S).detach().numpy())
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_blup(self, mod_lmm):
        expected = np.array(EXPECTED_RANDOM_REG["blup"])  # (n_levels, n_components)
        rand = mod_lmm.random[0]
        # pyreml: (n_components * n_levels,) -> reshape to (n_components, n_levels) -> T
        actual = rand.uhat.detach().numpy().reshape(rand.c, rand.L).T
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_residuals(self, mod_lmm):
        expected = EXPECTED_RANDOM_REG["residuals"]
        actual = mod_lmm.residual.table["residual"].tolist()
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_pev_diagonal_blocks(self, mod_lmm):
        expected = np.array(EXPECTED_RANDOM_REG["pev"])  # (2, 2, n_levels)
        rand = mod_lmm.random[0]
        pev = rand.PEV.detach().numpy()  # (k*c*L, k*c*L)
        c, L = rand.c, rand.L
        for i in range(L):
            # diagonal block for level i: rows/cols i, i+L (component-outer, level-inner)
            idx = [i + j * L for j in range(c)]
            block = pev[np.ix_(idx, idx)]
            np.testing.assert_allclose(block, expected[:, :, i], atol=2.5e-3)  # !!

    def test_tvals(self, mod_lmm):
            expected = EXPECTED_RANDOM_REG["tvals"]
            actual = mod_lmm.estimates["t"].tolist()
            np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_aic(self, mod_lmm):
            expected = EXPECTED_RANDOM_REG["aic"]
            actual = float(mod_lmm.AIC)
            np.testing.assert_allclose(actual, expected, atol=1e-3)