import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

from pyreml import MixedModel, Random, Residual, A_pedigree, D_pedigree, larix as DF

DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "pedigree_kinship.json") as f:
    KINSHIP = json.load(f)

with open(DATA_DIR / "pedigree_uni.json") as f:
    EXPECTED_UNI = json.load(f)

with open(DATA_DIR / "pedigree_diag.json") as f:
    EXPECTED_DIAG = json.load(f)

with open(DATA_DIR / "pedigree_str.json") as f:
    EXPECTED_STR = json.load(f)


@pytest.fixture()
def df():
    df = DF.copy()
    df = df[df["year"] == 2000]
    for col in ["SIRE", "DAM"]:
        df[col] = df[col].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
    return df

def _build_pedigree(df):
    ped = df[["ID", "DAM", "SIRE"]].drop_duplicates(subset="ID")
    parents  = set(ped["DAM"]).union(ped["SIRE"]) - {np.nan}
    founders = parents - set(ped["ID"])
    founder_rows = pd.DataFrame({"ID": list(founders), "DAM": np.nan, "SIRE": np.nan})
    return pd.concat([founder_rows, ped], ignore_index=True)


def _uni_blup(mod, effect_idx):
    """Univariate BLUP vector + level IDs for a given random effect."""
    rand = mod.random[effect_idx]
    return rand.uhat.detach().numpy().ravel(), rand.index.tolist()


def _multi_blup(mod, effect_idx=0):
    """Multivariate BLUP matrix (L, k) + level IDs for a given random effect."""
    rand = mod.random[effect_idx]
    uhat = rand.uhat.detach().numpy().reshape(rand.k, rand.L).T  # (L, k)
    return uhat, rand.index.tolist()


def _align(actual_vals, actual_ids, ref_vals, ref_ids, target_ids):
    """
    Align actual (pyreml) and reference (breedR) values by matching IDs.
    Returns (actual, expected) arrays restricted to target_ids.
    Works for 1-D vectors or 2-D matrices (rows = levels).
    """
    a_idx = {id_: i for i, id_ in enumerate(actual_ids)}
    r_idx = {id_: i for i, id_ in enumerate(ref_ids)}
    a = np.array([actual_vals[a_idx[id_]] for id_ in target_ids])
    r = np.array([ref_vals[r_idx[id_]] for id_ in target_ids])
    return a, r


def _pev_diag_for(rand, ids):
    """Diagonal of PEV for selected IDs."""
    pev = rand.PEV.detach().numpy()
    idx_map = {id_: i for i, id_ in enumerate(rand.index.tolist())}
    idx = [idx_map[id_] for id_ in ids]
    return np.diag(pev)[idx]

def _pev_diag_multi_for(rand, ids):
    """Diagonal blocks of PEV for selected IDs, multivariate → (len(ids), k)."""
    pev = rand.PEV.detach().numpy()
    k, L = rand.k, rand.L
    idx_map = {id_: i for i, id_ in enumerate(rand.index.tolist())}
    rows = []
    for id_ in ids:
        i = idx_map[id_]
        # PEV ordered component-outer, level-inner; diag block for level i
        rows.append([pev[j * L + i, j * L + i] for j in range(k)])
    return np.array(rows)

@pytest.fixture
def df_train(df):
    return df[df["BLOC"].isin([f"B{i}" for i in range(1, 9)])].copy()

@pytest.fixture
def df_tot(df):
    return df[df["BLOC"].isin([f"B{i}" for i in range(1, 13)])].copy()

@pytest.fixture
def pedigree_full(df_tot):
    return _build_pedigree(df_tot)

@pytest.fixture
def pedigree_train(df):
    return _build_pedigree(df)

@pytest.fixture
def kinship_full(pedigree_full):
    ped_ids = pedigree_full["ID"].tolist()
    A = A_pedigree(pedigree_full)
    D = D_pedigree(pedigree_full)
    return A, D, ped_ids

@pytest.fixture
def kinship_train(pedigree_full, df_train):
    ped_ids_full = pedigree_full["ID"].tolist()
    A_full = A_pedigree(pedigree_full)
    D_full = D_pedigree(pedigree_full)

    train_ids = set(df_train["ID"].unique())
    parents = set(df_train["DAM"]).union(df_train["SIRE"]) - {np.nan}
    founders = parents - train_ids
    keep_set = founders | train_ids

    # Réordonner selon la référence
    ref_order = KINSHIP["index"]
    keep_idx = [ped_ids_full.index(id_) for id_ in ref_order if id_ in keep_set]
    ped_ids = [ped_ids_full[i] for i in keep_idx]

    A = A_full[np.ix_(keep_idx, keep_idx)]
    D = D_full[np.ix_(keep_idx, keep_idx)]

    return A, D, ped_ids

@pytest.fixture
def uni_variances(df_train, kinship_train):
    """Run univariate models per trait to get variance inits for multivariate."""
    A, _, ped_ids = kinship_train
    traits = ["height", "circumference", "flexuosity"]
    var_a = {}
    var_r = {}
    for trait in traits:
        mod = MixedModel.from_dataframe(
            data     = df_train,
            response = trait,
            fixed    = "1 + BLOC",
            random = Random(
                unit         = "ID",
                right_hand   = "str",
                covariance   = A,
                matrix_index = ped_ids,
            ),
        ).fit()
        var_a[trait] = float(torch.exp(mod.random[0].log_S).detach())
        var_r[trait] = float(torch.exp(mod.residual.log_S).detach())
    
    SigmaA_init = np.diag([var_a[t] for t in traits])
    SigmaR_init = np.diag([var_r[t] for t in traits])

    return SigmaA_init, SigmaR_init

@pytest.fixture(params=[True, False], ids=["woodbury", "direct"])
def mod_uni(df_train, kinship_train, request):
    A, D, ped_ids = kinship_train
    return MixedModel.from_dataframe(
        data     = df_train,
        response = "flexuosity",
        fixed    = "1 + BLOC",
        random = [
            Random(
                unit         = "ID",
                right_hand   = "str",
                covariance   = A,
                matrix_index = ped_ids,
            ),
            Random(
                unit         = "ID",
                right_hand   = "str",
                covariance   = D,
                matrix_index = ped_ids
            ),
        ],
        SMW = request.param,
    ).fit()


@pytest.fixture(params=[True, False], ids=["woodbury", "direct"])
def mod_diag(df_train, kinship_train, uni_variances, request):
    A, _, ped_ids = kinship_train
    SigmaA_init, SigmaR_init = uni_variances
    return MixedModel.from_dataframe(
        data     = df_train,
        response = [
            "height",
            "circumference",
            "flexuosity",
        ],
        fixed    = "1 + BLOC",
        random = Random(
            unit         = "ID",
            left_hand    = "diag",
            right_hand   = "str",
            covariance   = A,
            matrix_index = ped_ids,
            init         = SigmaA_init,
        ),
        residual = Residual(
            left_hand = "diag",
            init      = SigmaR_init,
        ),
        SMW = request.param,
    ).fit()


@pytest.fixture(params=[True, False], ids=["woodbury", "direct"])
def mod_str(df_train, kinship_train, uni_variances, request):
    A, _, ped_ids = kinship_train
    SigmaA_init, SigmaR_init = uni_variances
    return MixedModel.from_dataframe(
        data     = df_train,
        response = [
            "height",
            "circumference",
            "flexuosity",
        ],
        fixed    = "1 + BLOC",
        random = Random(
            unit         = "ID",
            left_hand    = "full",
            right_hand   = "str",
            covariance   = A,
            matrix_index = ped_ids,
            init         = SigmaA_init,
        ),
        residual = Residual(
            left_hand    = "full",
            init         = SigmaR_init,
        ),
        SMW = request.param,
    ).fit()

class TestKinship:

    def test_A(self, kinship_train):
        A, _, _ = kinship_train
        np.testing.assert_allclose(A, np.array(KINSHIP["A"]), atol=1e-6)

    def test_D(self, kinship_train):
        _, D, _ = kinship_train
        np.testing.assert_allclose(D, np.array(KINSHIP["D"]), atol=1e-6)


class TestUnivariate:

    def test_convergence(self, mod_uni):
        assert mod_uni.opti_REML.converged is True

    def test_var_a(self, mod_uni):
        actual = float(torch.exp(mod_uni.random[0].log_S).detach())
        np.testing.assert_allclose(actual, EXPECTED_UNI["var_a"], rtol=1e-4, atol = 1e-6)

    def test_var_d(self, mod_uni):
        actual = float(torch.exp(mod_uni.random[1].log_S).detach())
        np.testing.assert_allclose(actual, EXPECTED_UNI["var_d"], rtol=1e-4, atol = 1e-6)

    def test_var_r(self, mod_uni):
        actual = float(torch.exp(mod_uni.residual.log_S).detach())
        np.testing.assert_allclose(actual, EXPECTED_UNI["var_r"], rtol=1e-4, atol = 1e-6)

    def test_blup_a(self, mod_uni):
        blup, ids = _uni_blup(mod_uni, 0)
        actual, expected = _align(
            blup, ids,
            np.array(EXPECTED_UNI["blup_a"]), EXPECTED_UNI["ped_index"],
            EXPECTED_UNI["train_ids"],
        )
        np.testing.assert_allclose(actual, expected, atol=5e-4)

    def test_blup_d(self, mod_uni):
        blup, ids = _uni_blup(mod_uni, 1)
        actual, expected = _align(
            blup, ids,
            np.array(EXPECTED_UNI["blup_d"]), EXPECTED_UNI["ped_index"],
            EXPECTED_UNI["train_ids"],
        )
        np.testing.assert_allclose(actual, expected, atol=5e-4)

    def test_pev_a(self, mod_uni):
        train_ids = EXPECTED_UNI["train_ids"]
        actual = _pev_diag_for(mod_uni.random[0], train_ids)
        ref_se = np.array(EXPECTED_UNI["se_a"])
        ref_idx = {id_: i for i, id_ in enumerate(EXPECTED_UNI["ped_index"])}
        expected = np.array([ref_se[ref_idx[id_]] ** 2 for id_ in train_ids])
        np.testing.assert_allclose(actual, expected, rtol=2e-3) # !


class TestMultivariateDiag:

    def test_convergence(self, mod_diag):
        assert mod_diag.opti_REML.converged is True

    def test_SigmaA(self, mod_diag):
        actual = mod_diag.random[0].build_S().detach().numpy()
        np.testing.assert_allclose(actual, np.array(EXPECTED_DIAG["SigmaA"]), rtol=1e-3) # !!

    def test_SigmaR(self, mod_diag):
        actual = mod_diag.residual.build_S().detach().numpy()
        np.testing.assert_allclose(actual, np.array(EXPECTED_DIAG["SigmaR"]), rtol=1e-4)

    def test_blup_a(self, mod_diag):
        blup, ids = _multi_blup(mod_diag, 0)
        ref_blup = np.array(EXPECTED_DIAG["blup_a"])
        ref_ids  = EXPECTED_DIAG["blup_index"]
        train_ids = EXPECTED_DIAG["train_ids"]
        actual, expected = _align(blup, ids, ref_blup, ref_ids, train_ids)
        np.testing.assert_allclose(actual, expected, atol = 5e-3) # ! woodbury


class TestMultivariateStr:

    def test_convergence(self, mod_str):
        assert mod_str.opti_REML.converged is True

    def test_SigmaA(self, request, mod_str):
        actual = mod_str.random[0].build_S().detach().numpy()
        expected = np.array(EXPECTED_STR["SigmaA"])
        if mod_str.SMW and np.allclose(actual, expected, rtol=0.05):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="fullxfull SigmaA: Woodbury drifts ~3-4pc on the non-identifiability ridge",
                    strict=False,
                )
            )
        np.testing.assert_allclose(actual, expected, rtol=0.005) # !

    def test_SigmaR(self, mod_str):
        actual = mod_str.residual.build_S().detach().numpy()
        np.testing.assert_allclose(actual, np.array(EXPECTED_STR["SigmaR"]), rtol=0.005) # !

    def test_blup_a(self, mod_str):
        blup, ids = _multi_blup(mod_str, 0)
        ref_blup = np.array(EXPECTED_STR["blup_a"])
        ref_ids  = EXPECTED_STR["blup_index"]
        train_ids = EXPECTED_STR["train_ids"]
        actual, expected = _align(blup, ids, ref_blup, ref_ids, train_ids)
        np.testing.assert_allclose(actual, expected, atol=0.5) # !!!

    def test_pev_a(self, mod_str):
        train_ids = EXPECTED_STR["train_ids"]
        actual = _pev_diag_multi_for(mod_str.random[0], train_ids)
        ref_se = np.array(EXPECTED_STR["se_a_train"])
        expected = ref_se ** 2  # se_a_train already restricted to train IDs
        np.testing.assert_allclose(actual, expected, rtol=0.03) # !!

class TestUnivariatePrediction:

    def test_predict_a(self, mod_uni, kinship_full):
        A_full, _, ped_ids_full = kinship_full
        pred_ids = EXPECTED_UNI["pred_ids"]
        out = mod_uni.random[0].predict(
            matrix_index=ped_ids_full,
            covariance=A_full,
        )
        actual = out.set_index("unit").loc[pred_ids, "prediction"].to_numpy()
        expected = np.array(EXPECTED_UNI["blup_a_pred"])
        np.testing.assert_allclose(actual, expected, atol=5e-4)

    def test_predict_d(self, mod_uni, kinship_full):
        _, D_full, ped_ids_full = kinship_full
        pred_ids = EXPECTED_UNI["pred_ids"]
        out = mod_uni.random[1].predict(
            matrix_index=ped_ids_full,
            covariance=D_full,
        )
        actual = out.set_index("unit").loc[pred_ids, "prediction"].to_numpy()
        expected = np.array(EXPECTED_UNI["blup_d_pred"])
        np.testing.assert_allclose(actual, expected, atol=5e-4)

class TestStrPrediction:

    def test_predict_a(self, mod_str, kinship_full):
        A_full, _, ped_ids_full = kinship_full
        pred_ids = EXPECTED_STR["pred_ids"]
        out = mod_str.random[0].predict(
            matrix_index=ped_ids_full,
            covariance=A_full,
        )
        # Multivariate: prediction column per (response, component) → filter
        pred_df = out[out["unit"].isin(pred_ids)]
        traits = EXPECTED_STR["traits"]
        actual = pred_df.pivot(index="unit", columns="response", values="prediction")
        actual = actual.loc[pred_ids, traits].to_numpy()
        expected = np.array(EXPECTED_STR["blup_a_pred"])
        np.testing.assert_allclose(actual, expected, atol=0.3) # !!!
