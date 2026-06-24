import json
import os

import numpy as np
import pandas as pd
import pytest

from pyreml import (
    MixedModel,
    Random,
    Residual,
    A_pedigree,
    prepare_pedigree,
    larix,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ATOL = 1e-6
RTOL = 1e-6

def _ref(name):
    with open(os.path.join(DATA, f"doc_{name}.json")) as f:
        return json.load(f)


def _close(got, ref, err=""):
    """Tight array comparison against the frozen reference."""
    np.testing.assert_allclose(
        np.asarray(got, dtype=float),
        np.asarray(ref, dtype=float),
        atol=ATOL,
        rtol=RTOL,
        err_msg=err,
    )


def _assert_table(got_df, ref_records, numeric_cols, label_cols):
    """
    Positional, machine-precision comparison of a result table against its
    frozen records. Positional (not merged) on purpose: row order is itself
    part of the non-regression contract, so a reordering must fail.
    """
    ref_df = pd.DataFrame(ref_records)
    assert len(got_df) == len(ref_df), (
        f"row count {len(got_df)} != {len(ref_df)}"
    )
    got = got_df.reset_index(drop=True)
    ref = ref_df.reset_index(drop=True)

    for c in label_cols:
        assert list(map(str, got[c])) == list(map(str, ref[c])), (
            f"label column '{c}' differs"
        )
    for c in numeric_cols:
        _close(
            got[c].to_numpy(dtype=float),
            ref[c].to_numpy(dtype=float),
            err=f"numeric column '{c}' differs",
        )

@pytest.fixture(scope="session")
def spatial():
    df = larix.copy()
    df = df[df["year"] == 2000].copy()
    df["ID"] = np.arange(len(df))

    coords = df[["X", "Y"]].to_numpy()
    D_train = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :],
        axis=-1,
    )

    mod = MixedModel.from_dataframe(
        data     = df,
        response = "height",
        fixed    = "1",
        random   = Random(
            unit         = "ID",
            right_hand   = "ar",
            distance     = D_train,
            matrix_index = df["ID"].tolist(),
        ),
    ).fit()

    # coarse grid (step 10): must match the frozen reference produced by doc.py
    gx = np.arange(coords[:, 0].min() - 20, coords[:, 0].max() + 20, 10)
    gy = np.arange(coords[:, 1].min() - 20, coords[:, 1].max() + 20, 10)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    grid_ids = np.arange(len(df), len(df) + len(grid))
    all_coords = np.vstack([coords, grid])
    D_full = np.linalg.norm(
        all_coords[:, None, :] - all_coords[None, :, :],
        axis=-1,
    )

    pred = mod.random[0].predict(
        matrix_index = df["ID"].tolist() + grid_ids.tolist(),
        distance     = D_full,
    )
    return {"mod": mod, "pred": pred, "n_train": len(df), "n_grid": len(grid)}


@pytest.fixture(scope="session")
def fa():
    df = larix.copy()
    traits = ["height", "circumference", "flexuosity"]
    df = df[df["year"].isin([2000, 2014])]

    long = df.melt(
        id_vars=["ID", "DAM", "SIRE", "BLOC", "year"],
        value_vars=traits,
        var_name="trait",
        value_name="value",
    )
    long["resp"] = long["trait"] + "_" + long["year"].astype(str)
    wide = (
        long.pivot_table(
            index=["ID", "DAM", "SIRE", "BLOC"],
            columns="resp",
            values="value",
        )
        .dropna(axis=1, how="all")
        .reset_index()
    )
    responses = [c for c in wide.columns if c not in ("ID", "DAM", "SIRE", "BLOC")]

    ped = prepare_pedigree(wide[["ID", "DAM", "SIRE"]])
    A = A_pedigree(ped)
    P = wide[responses].cov().to_numpy()

    model = MixedModel.from_dataframe(
        data     = wide,
        response = responses,
        fixed    = "1",
        random   = Random(
            unit         = "ID",
            left_hand    = "fa",
            n_axes       = 2,
            right_hand   = "str",
            covariance   = A,
            matrix_index = ped["id"].tolist(),
            init         = P / 2,
            jitter       = 1e-6,
        ),
        residual = Residual(
            left_hand    = "fa",
            n_axes       = 2,
            right_hand   = "iid",
            init         = P / 2,
            jitter       = 1e-6,
        ),
    ).fit()
    return {"model": model, "responses": responses}


@pytest.fixture(scope="session")
def regression():
    df = larix.copy()
    df["year"] = df["year"] - df["year"].min()

    ped = prepare_pedigree(df[["ID", "DAM", "SIRE"]])
    K = A_pedigree(ped)

    mod = MixedModel.from_dataframe(
        data     = df,
        response = "height",
        fixed    = "1 + year",
        random   = Random(
            unit         = "ID",
            formula      = "1 + year",
            left_hand    = "full",
            right_hand   = "str",
            covariance   = K,
            matrix_index = ped["id"].tolist(),
            jitter       = 1e-6,
        ),
        residual = Residual(
            left_hand = "iid",
            right_hand = "het",
            het_formula = "C(year)"
        ),
    ).fit()
    return {"mod": mod}

class TestSpatial:
    def test_dimensions(self, spatial):
        ref = _ref("spatial")
        assert spatial["n_train"] == ref["n_train"]
        assert spatial["n_grid"] == ref["n_grid"]

    def test_estimates(self, spatial):
        ref = _ref("spatial")
        _assert_table(
            spatial["mod"].estimates,
            ref["estimates"],
            numeric_cols=["estimate"],
            label_cols=["response", "term"],
        )

    def test_ar_parameter(self, spatial):
        ref = _ref("spatial")
        _close(spatial["mod"].random[0].variance["metadata"]["ar"], ref["ar"], "ar")

    def test_additive_variance(self, spatial):
        ref = _ref("spatial")
        _close(
            float(spatial["mod"].random[0].build_S().detach()),
            ref["var_additive"],
            "var_additive",
        )

    def test_blup(self, spatial):
        ref = _ref("spatial")
        _assert_table(
            spatial["mod"].random[0].table,
            ref["blup"],
            numeric_cols=["prediction"],
            label_cols=["unit", "response", "component"],
        )

    def test_residuals(self, spatial):
        ref = _ref("spatial")
        _assert_table(
            spatial["mod"].residual.table,
            ref["residuals"],
            numeric_cols=["residual"],
            label_cols=["observation", "response"],
        )

    def test_kriging_prediction(self, spatial):
        ref = _ref("spatial")
        _assert_table(
            spatial["pred"],
            ref["prediction"],
            numeric_cols=["prediction"],
            label_cols=["unit", "response", "component"],
        )

class TestRegression:
    def test_estimates(self, regression):
        ref = _ref("regression")
        _assert_table(
            regression["mod"].estimates,
            ref["estimates"],
            numeric_cols=["estimate"],
            label_cols=["response", "term"],
        )

    def test_variance(self, regression):
        ref = _ref("regression")
        var = regression["mod"].random[0].variance
        _close(var["sigma"], ref["S_random"], "S_random")
        assert var["metadata"]["labels"] == ref["variance_labels"]

    def test_blup(self, regression):
        ref = _ref("regression")
        _assert_table(
            regression["mod"].random[0].table,
            ref["blup"],
            numeric_cols=["prediction"],
            label_cols=["unit", "response", "component"],
        )

    def test_residuals(self, regression):
        ref = _ref("regression")
        _assert_table(
            regression["mod"].residual.table,
            ref["residuals"],
            numeric_cols=["residual"],
            label_cols=["observation", "response"],
        )

class TestFA:
    def test_responses(self, fa):
        ref = _ref("fa")
        assert list(fa["responses"]) == ref["responses"]

    def test_estimates(self, fa):
        ref = _ref("fa")
        _assert_table(
            fa["model"].estimates,
            ref["estimates"],
            numeric_cols=["estimate"],
            label_cols=["response", "term"],
        )

    def test_random_fa_structure(self, fa):
        ref = _ref("fa")
        var = fa["model"].random[0].variance
        meta = var["metadata"]["fa"]
        _close(var["sigma"], ref["S_random"], "S_random")
        _close(meta["Q"], ref["Q_random"], "Q_random")
        _close(meta["Lambda"], ref["Lambda_random"], "Lambda_random")
        _close(meta["Psi"], ref["Psi_random"], "Psi_random")

    def test_residual_fa_structure(self, fa):
        ref = _ref("fa")
        var = fa["model"].residual.variance
        meta = var["metadata"]["fa"]
        _close(var["sigma"], ref["S_residual"], "S_residual")
        _close(meta["Q"], ref["Q_residual"], "Q_residual")
        _close(meta["Lambda"], ref["Lambda_residual"], "Lambda_residual")
        _close(meta["Psi"], ref["Psi_residual"], "Psi_residual")

    def test_blup(self, fa):
        ref = _ref("fa")
        _assert_table(
            fa["model"].random[0].table,
            ref["blup"],
            numeric_cols=["prediction"],
            label_cols=["unit", "response", "component"],
        )

    def test_residuals(self, fa):
        ref = _ref("fa")
        _assert_table(
            fa["model"].residual.table,
            ref["residuals"],
            numeric_cols=["residual"],
            label_cols=["observation", "response"],
        )
