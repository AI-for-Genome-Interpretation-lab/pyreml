import json
import os

import numpy as np
import pytest
import pandas as pd

from pyreml import MixedModel, Random, Residual, A_genomic

DEVICE = "cuda"

HERE = os.path.dirname(os.path.abspath(__file__))
REF_JSON = os.path.join(HERE, "data", "genomic_sim.json")
REF_BLKR_JSON = os.path.join(HERE, "data", "genomic_blkr.json")

P = 5
N_AXES = 2

RESP = {0: [0, 1], 1: [2, 3]}
FORM = {"Intercept": [0, 2], "x": [1, 3]}
RESPONSES = ["y_0", "y_1"]
TERMS = ["Intercept", "x"]

MODELS = ["bl_resp", "bl_form", "kr_resp", "kr_form"]

@pytest.fixture(scope="session")
def ref():
    with open(REF_JSON) as f:
        return json.load(f)

@pytest.fixture(scope="session")
def sim(ref):
    """Reconstruct everything the tests need from the frozen reference."""
    id_index = ref["index"]

    df_long = pd.DataFrame(ref["df_long"])
    df_long["envt"] = df_long["envt"].astype("category")

    df_cols = pd.DataFrame(ref["df_cols"])

    return {
        "id_index": id_index,
        "obs_ids": set(df_long["ID"].unique()),
        "A": np.array(ref["A"]),
        "X": np.array(ref["X"]),
        "K": np.array(ref["K_genomic"]),
        "K_genomic": np.array(ref["K_genomic"]),
        "means": np.array(ref["means"]),
        "Sigma_A": np.array(ref["Sigma_A"]),
        "Sigma_R": np.array(ref["Sigma_R"]),
        "Lambda": np.array(ref["Lambda"]),
        "Q": np.array(ref["Q"]),
        "rel_inertia": np.array(ref["rel_inertia"]),
        "u_true": np.array(ref["u_true"]),   # (n, p)
        "var_additive_het": ref["var_additive_het"],
        "df_long": df_long,
        "df_cols": df_cols,
    }


@pytest.fixture(scope="session")
def G(sim):
    return A_genomic(sim["X"], shrink=True)

@pytest.fixture(scope="session", params=[True, False], ids=["woodbury", "direct"])
def fitted_het(sim, G, request):
    return MixedModel.from_dataframe(
        data       = sim["df_long"],
        response   = "y",
        fixed      = "1 + envt",
        random = Random(
            unit         = "ID",
            right_hand   = "str",
            covariance   = G,
            matrix_index = sim["id_index"],
        ),
        residual = Residual(
            right_hand   = "het",
            het_formula  = "C(envt)",
        ),
        SMW = request.param,
        device = DEVICE,
    ).fit()

@pytest.fixture(scope="session", params=[True, False], ids=["woodbury", "direct"])
def fitted_fa(sim, G, request):
    return MixedModel.from_dataframe(
        data     = sim["df_cols"],
        response = [f"y_{envt}" for envt in range(P)],
        fixed    = "1",
        random = Random(
            unit         = "ID",
            left_hand    = "fa",
            right_hand   = "str",
            covariance   = G,
            matrix_index = sim["id_index"],
            n_axes       = N_AXES,
        ),
        residual = Residual(
            left_hand    = "diag",
        ),
        SMW = request.param,
        device = DEVICE,
    ).fit()

@pytest.fixture(scope="session")
def ref_blkr():
    with open(REF_BLKR_JSON) as f:
        return json.load(f)

@pytest.fixture(scope="session")
def sim_blkr(ref_blkr):
    df_cols = pd.DataFrame(ref_blkr["df_cols"])
    return {
        "id_index": ref_blkr["id_index"],
        "obs_ids": set(df_cols["ID"].unique()),
        "Sigma_A": np.array(ref_blkr["Sigma_A"]),
        "Sigma_R": np.array(ref_blkr["Sigma_R"]),
        "true_beta": {r: np.array(v) for r, v in ref_blkr["true_beta"].items()},
        "u_true": np.array(ref_blkr["u_true"]),          # (n, 4)
        "df_cols": pd.DataFrame(df_cols),
    }

@pytest.fixture(scope="session", params=[
    (lh, smw) for lh in MODELS for smw in (True, False)
], ids=lambda p: f"{p[0]}-{'woodbury' if p[1] else 'direct'}")
def fitted_blkr(sim_blkr, G, request):
    lh, smw = request.param
    mod = MixedModel.from_dataframe(
        data     = sim_blkr["df_cols"],
        response = RESPONSES,
        fixed    = "1 + x",
        random = Random(
            unit         = "ID",
            formula      = "1 + x",
            left_hand    = lh,
            right_hand   = "str",
            covariance   = G,
            matrix_index = sim_blkr["id_index"],
        ),
        residual = Residual(left_hand="diag"),
        SMW = smw,
        device = DEVICE,
    ).fit()
    return lh, mod

def _abs_corr(a, b):
    """|Pearson correlation| -- sign-agnostic, for factor axes / eigenvectors."""
    return abs(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])

class TestAGenomic:
    def test_matches_reference(self, G, sim):
        np.testing.assert_allclose(G, sim["K_genomic"], atol=1e-10)

class TestHet:
    def test_environment_means(self, fitted_het, sim):
        raw = fitted_het.estimates["estimate"].tolist()
        intercept = raw[0]
        mean_hat = [intercept] + [intercept + c for c in raw[1:]]
        true_means = sim["means"]
        for envt in range(P):
            err = abs(mean_hat[envt] - true_means[envt])
            assert err <= 2.0

    def test_additive_variance_finite_positive(self, fitted_het):
        """The het additive variance is the loose point of the reference
        (it conflates one shared effect across 5 environments). We only assert
        it stays finite and strictly positive; the true value is documented in
        the reference but not bounded here."""
        var_a = float(fitted_het.random[0].build_S().detach())
        assert np.isfinite(var_a) and var_a > 0.0

    def test_residual_variances(self, fitted_het, sim):
        fitted_het.residual.format_variance()
        var = fitted_het.residual.variance

        sigma_base = float(var["sigma"])          # reference modality (env 0)
        het = var["metadata"]["het"]              # [{column, h}, ...] non-reference

        # env 0 = reference (carries the scale); env k = sigma_base * h
        sigma_r_hat = {0: sigma_base}
        for entry in het:
            k = int(entry["column"].split("[T.")[1].rstrip("]"))   # "C(envt)[T.k]" -> k
            sigma_r_hat[k] = sigma_base * entry["h"]

        true_r = sim["Sigma_R"]
        for envt in range(P):
            rel_err = abs(sigma_r_hat[envt] - true_r[envt]) / true_r[envt]
            assert rel_err <= 0.5

    def test_blup_accuracy(self, fitted_het, sim):
        blup = fitted_het.random[0].table[["unit", "prediction"]]
        u_true = pd.DataFrame(
            {"unit": sim["id_index"], "u_true": sim["u_true"][:, 0]}
        )
        cmp = blup.merge(u_true, on="unit")
        cmp = cmp[cmp["unit"].isin(sim["obs_ids"])]
        acc = np.corrcoef(cmp["prediction"], cmp["u_true"])[0, 1]
        assert acc >= 0.99

    def test_r2(self, fitted_het, sim):
        resid_tab = fitted_het.residual.table
        e = resid_tab["residual"].to_numpy()
        y_obs = sim["df_long"]["y"].to_numpy()
        assert len(e) == len(y_obs)
        ss_res = np.sum(e ** 2)
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 >= 0.95

class TestFA:
    def test_environment_means(self, fitted_fa, sim):
        est = fitted_fa.estimates
        mean_fa = (
            est[est["term"] == "Intercept"]
            .sort_values("response")["estimate"]
            .tolist()
        )
        true_means = sim["means"]
        for envt in range(P):
            err = abs(mean_fa[envt] - true_means[envt])
            assert err <= 2

    def test_factor_axes_concordance(self, fitted_fa, sim):
        """Estimated factor axes vs simulated eigenvectors (sign-agnostic).
        Axes are assumed ordered by decreasing Lambda on both sides."""
        fa = fitted_fa.random[0].variance["metadata"]["fa"]
        Q_hat = np.asarray(fa["Q"])
        Q_true = sim["Q"]
        for ax in range(N_AXES):
            r = _abs_corr(Q_hat[:, ax], Q_true[:, ax])
            assert r >= 0.85


    def test_fa_Sigma(self, fitted_fa):
        """FA metadata must reconstruct the reported natural covariance."""
        fitted_fa.random[0].format_variance()

        fa = fitted_fa.random[0].variance["metadata"]["fa"]
        Q = np.asarray(fa["Q"])
        Lambda = np.asarray(fa["Lambda"])
        Psi = np.asarray(fa["Psi"])

        S_from_metadata = Q @ np.diag(Lambda) @ Q.T + np.diag(Psi)
        S_from_model = fitted_fa.random[0].build_S().detach().cpu().numpy()

        np.testing.assert_allclose(
            S_from_metadata,
            S_from_model,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_relative_inertia(self, fitted_fa, sim):
        """Per-axis relative inertia: estimated share (on estimated total)
        vs true share (on true total). Axes 0 and 1 only."""
        fa = fitted_fa.random[0].variance["metadata"]["fa"]
        Lambda_hat = np.asarray(fa["Lambda"])
        S_hat = fitted_fa.random[0].build_S().detach().numpy()
        inertia_hat = np.trace(S_hat)

        true_rel = sim["rel_inertia"]
        for ax in range(N_AXES):
            rel_hat = Lambda_hat[ax] / inertia_hat
            err = abs(rel_hat - true_rel[ax])
            assert err <= 0.25

    def test_residual_variances(self, fitted_fa, sim):
        fitted_fa.residual.format_variance()
        S_r_hat = np.diag(fitted_fa.residual.variance["sigma"])
        true_r = sim["Sigma_R"]
        for envt in range(P):
            rel_err = abs(S_r_hat[envt] - true_r[envt]) / true_r[envt]
            assert rel_err <= 0.90, (
                f"env {envt}: residual rel err {rel_err:.3f} > 0.90"
            )

    def test_blup_accuracy_per_environment(self, fitted_fa, sim):
        tab = fitted_fa.random[0].table
        u_true = sim["u_true"]  # (n, P)
        u_true_df = pd.DataFrame(
            {
                "unit": sim["id_index"],
                **{f"y_{e}": u_true[:, e] for e in range(P)},
            }
        )
        for envt in range(P):
            resp = f"y_{envt}"
            pred = tab[tab["response"] == resp][["unit", "prediction"]]
            cmp = pred.merge(u_true_df[["unit", resp]], on="unit")
            cmp = cmp[cmp["unit"].isin(sim["obs_ids"])]
            acc = np.corrcoef(cmp["prediction"], cmp[resp])[0, 1]
            assert acc >= 0.75

    def test_r2(self, fitted_fa, sim):
        resid_tab = fitted_fa.residual.table
        e = resid_tab["residual"].to_numpy()
        y_stacked = np.concatenate(
            [sim["df_cols"][f"y_{e}"].dropna().to_numpy() for e in range(P)]
        )
        assert len(e) == len(y_stacked)
        ss_res = np.sum(e ** 2)
        ss_tot = np.sum((y_stacked - y_stacked.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 >= 0.95

def _block(S, rows, cols):
    return S[np.ix_(rows, cols)]


class TestBlKr:
    def test_structural_zeros(self, fitted_blkr):
        """Off-block entries constrained to zero must be exactly zero (wiring check)."""
        lh, mod = fitted_blkr
        S = mod.random[0].build_S().detach().numpy()
        if lh in ("bl_resp", "kr_resp"):
            cross = _block(S, RESP[0], RESP[1])      # cross-response block
        else:
            cross = _block(S, FORM["Intercept"], FORM["x"])  # intercept–slope block
        assert np.abs(cross).max() <= 1e-10

    def test_variances_present_and_plausible(self, fitted_blkr, sim_blkr):
        lh, mod = fitted_blkr
        S = mod.random[0].build_S().detach().numpy()
        true_var = np.diag(sim_blkr["Sigma_A"])
        for i in range(S.shape[0]):
            v = S[i, i]
            assert np.isfinite(v) and v > 0.0
            assert 0.4 * true_var[i] <= v <= 2.0 * true_var[i], (
                f"{lh} diag[{i}]={v:.3f}"
            )
        # proportionality: only for kr structures, read the stored coefficient
        if lh in ("kr_resp", "kr_form"):
            mod.random[0].format_variance()
            kr = mod.random[0].variance["metadata"]["kr"]
            ratios = [r["alpha" if lh == "kr_resp" else "omega"] for r in kr["ratios"]]
            # le ratio bloc/bloc doit égaler le coef stocké (proportionnalité exacte)
            if lh == "kr_resp":
                block_ratio = _block(S, RESP[1], RESP[1]) / _block(S, RESP[0], RESP[0])
            else:
                block_ratio = _block(S, FORM["x"], FORM["x"]) / _block(S, FORM["Intercept"], FORM["Intercept"])
            cv = block_ratio.std() / abs(block_ratio.mean())
            assert cv <= 1e-4, f"{lh}: non-constant ratio, cv={cv:.2e}"
            np.testing.assert_allclose(block_ratio.mean(), ratios[0], rtol=1e-4)

    def test_residual_variances(self, fitted_blkr, sim_blkr):
        lh, mod = fitted_blkr
        mod.residual.format_variance()
        S_r = np.diag(mod.residual.variance["sigma"])
        true_r = sim_blkr["Sigma_R"]
        for j, resp in enumerate(RESPONSES):
            rel_err = abs(S_r[j] - true_r[j]) / true_r[j]
            assert rel_err <= 0.30, f"{lh} {resp}: residual relative error {rel_err:.3f}"

    def test_environment_means(self, fitted_blkr, sim_blkr):
        """Beta coefficients: intercept and slope per response, with a broad absolute tolerance."""
        lh, mod = fitted_blkr
        est = mod.estimates
        for resp in RESPONSES:
            true_int, true_slope = sim_blkr["true_beta"][resp]
            sub = est[est["response"] == resp].set_index("term")["estimate"]
            assert abs(sub["Intercept"] - true_int) <= 3.0
            assert abs(sub["x"] - true_slope) <= 1.5

    def test_blup_accuracy(self, fitted_blkr, sim_blkr):
        """BLUP accuracy for each (response, component) pair."""
        lh, mod = fitted_blkr
        u_true = sim_blkr["u_true"]   # (n, 4), component order
        col = {("y_0", "Intercept"): 0, ("y_0", "x"): 1,
               ("y_1", "Intercept"): 2, ("y_1", "x"): 3}
        tab = mod.random[0].table
        u_df = pd.DataFrame({"unit": sim_blkr["id_index"]})
        for (resp, comp), c in col.items():
            u_df[f"{resp}|{comp}"] = u_true[:, c]
        for resp in RESPONSES:
            for comp in TERMS:
                pred = tab[(tab["response"] == resp) & (tab["component"] == comp)][["unit", "prediction"]]
                cmp = pred.merge(u_df[["unit", f"{resp}|{comp}"]], on="unit")
                cmp = cmp[cmp["unit"].isin(sim_blkr["obs_ids"])]
                acc = np.corrcoef(cmp["prediction"], cmp[f"{resp}|{comp}"])[0, 1]
                assert acc >= 0.85, f"{lh} {resp} {comp}: acc={acc:.4f}"

    def test_r2(self, fitted_blkr, sim_blkr):
        lh, mod = fitted_blkr
        resid_tab = mod.residual.table
        for resp in RESPONSES:
            e = resid_tab[resid_tab["response"] == resp]["residual"].to_numpy()
            yv = sim_blkr["df_cols"][resp].to_numpy()
            r2 = 1.0 - np.sum(e ** 2) / np.sum((yv - yv.mean()) ** 2)
            assert r2 >= 0.80, f"{lh} {resp}: R²={r2:.4f}"