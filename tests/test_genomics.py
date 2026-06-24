import json
import os

import numpy as np
import pytest
import pandas as pd

from pyreml import MixedModel, Random, Residual, A_genomic

HERE = os.path.dirname(os.path.abspath(__file__))
REF_JSON = os.path.join(HERE, "data", "genomic_sim.json")

P = 5
N_AXES = 2

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
    ).fit()

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