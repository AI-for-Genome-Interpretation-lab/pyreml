"""
Low-level constructor, cross-checked against the existing references.

Each model is laid out by from_dataframe (dense Z forced), then refitted
through the low-level constructor with hand-written variance methods.
The solving path is forced by the method supplied: varmeth alone routes
to the direct solver, varmeth_inv alone to Woodbury (SMW).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pyreml import MixedModel, Random, A_pedigree, D_pedigree, larix as DF

DEVICE = "cpu"
DTYPE = "mixed"
DATA_DIR = Path(__file__).parent / "data"
PATHS = ["direct", "woodbury"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _load(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)


def _dense_design(**kwargs):
    """High-level build used as a layout only: y, X, dense Z, level order."""
    return MixedModel.from_dataframe(
        **kwargs,
        SMW=False,
        structured_forward=False,
        analytic_backward=False,
        device=DEVICE,
    )


def _leaf(value):
    """Trainable scalar or tensor, created as a double leaf on DEVICE."""
    return torch.tensor(value, dtype=torch.double, device=DEVICE, requires_grad=True)


def _fit_low_level(design, params, varmeth, varmeth_inv, path):
    model = MixedModel(
        y=design.y,
        X=design.X,
        Z=design.Z,
        params=params,
        varmeth=varmeth if path == "direct" else None,
        varmeth_inv=varmeth_inv if path == "woodbury" else None,
        device=DEVICE,
    )
    return model.fit(DTYPE, verbose=False)


def _np(t):
    return t.detach().cpu().numpy()


def _assert_routing(fit):
    model, path = fit["model"], fit["path"]
    assert model.SMW is (path == "woodbury")


# --------------------------------------------------------------------------- #
# Pedigree: additive + dominance
# --------------------------------------------------------------------------- #
EXPECTED_PED = _load("pedigree_uni.json")


def _pedigree_data():
    """Year-2000 larix, training blocs B1-B8, kinships on founders + train IDs."""
    df = DF[DF["year"] == 2000].copy()
    for col in ("SIRE", "DAM"):
        df[col] = df[col].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)

    df_tot = df[df["BLOC"].isin([f"B{i}" for i in range(1, 13)])]
    df_train = df[df["BLOC"].isin([f"B{i}" for i in range(1, 9)])].copy()

    ped = df_tot[["ID", "DAM", "SIRE"]].drop_duplicates(subset="ID")
    parents = set(ped["DAM"]).union(ped["SIRE"]) - {np.nan}
    founders = parents - set(ped["ID"])
    ped_full = pd.concat(
        [pd.DataFrame({"ID": list(founders), "DAM": np.nan, "SIRE": np.nan}), ped],
        ignore_index=True,
    )
    ids_full = ped_full["ID"].tolist()
    A_full, D_full = A_pedigree(ped_full), D_pedigree(ped_full)

    train_ids = set(df_train["ID"].unique())
    train_parents = set(df_train["DAM"]).union(df_train["SIRE"]) - {np.nan}
    keep = train_ids | (train_parents - train_ids)
    idx = [i for i, u in enumerate(ids_full) if u in keep]

    ped_ids = [ids_full[i] for i in idx]
    return df_train, A_full[np.ix_(idx, idx)], D_full[np.ix_(idx, idx)], ped_ids


def _reorder(K, ped_ids, index):
    """Permute a kinship from pedigree order to the Z column (level) order."""
    pos = {u: i for i, u in enumerate(ped_ids)}
    idx = [pos[u] for u in index]
    return K[np.ix_(idx, idx)]


@pytest.fixture(scope="module", params=PATHS)
def ped_fit(request):
    df_train, A, D, ped_ids = _pedigree_data()
    design = _dense_design(
        data=df_train,
        response="flexuosity",
        fixed="1 + BLOC",
        random=[
            Random(unit="ID", right_hand="str", covariance=A, matrix_index=ped_ids),
            Random(unit="ID", right_hand="str", covariance=D, matrix_index=ped_ids),
        ],
    )
    index = design.random[0].index.tolist()
    L = len(index)

    A_t = torch.as_tensor(_reorder(A, ped_ids, index), dtype=torch.double, device=DEVICE)
    D_t = torch.as_tensor(_reorder(D, ped_ids, index), dtype=torch.double, device=DEVICE)
    Ainv, Dinv = torch.linalg.inv(A_t), torch.linalg.inv(D_t)
    logdet_A, logdet_D = torch.logdet(A_t), torch.logdet(D_t)

    log_v0 = float(np.log(design.y.var().item()))
    log_sa, log_sd, log_se = _leaf(log_v0), _leaf(log_v0), _leaf(log_v0)

    def varmeth(self):
        G = torch.block_diag(
            torch.exp(log_sa) * A_t.to(self.dtype),
            torch.exp(log_sd) * D_t.to(self.dtype),
        )
        R = torch.exp(log_se) * torch.eye(self.n, dtype=self.dtype, device=self.device)
        return G, R

    def varmeth_inv(self):
        Ginv = torch.block_diag(
            torch.exp(-log_sa) * Ainv.to(self.dtype),
            torch.exp(-log_sd) * Dinv.to(self.dtype),
        )
        logdet_G = L * (log_sa + log_sd) + logdet_A + logdet_D
        Rinv = torch.exp(-log_se) * torch.eye(self.n, dtype=self.dtype, device=self.device)
        logdet_R = self.n * log_se
        return Ginv, Rinv, logdet_G, logdet_R

    model = _fit_low_level(design, [log_sa, log_sd, log_se], varmeth, varmeth_inv, request.param)
    return dict(model=model, path=request.param, index=index, L=L,
                var=[log_sa, log_sd, log_se])


def _ped_aligned(values, index, ref_key):
    """Restrict (actual, expected) to the train IDs, matched by ID."""
    a_pos = {u: i for i, u in enumerate(index)}
    r_pos = {u: i for i, u in enumerate(EXPECTED_PED["ped_index"])}
    ref = np.asarray(EXPECTED_PED[ref_key])
    ids = EXPECTED_PED["train_ids"]
    return (
        np.array([values[a_pos[u]] for u in ids]),
        np.array([ref[r_pos[u]] for u in ids]),
    )


class TestLowLevelPedigree:

    def test_routing(self, ped_fit):
        _assert_routing(ped_fit)

    def test_convergence(self, ped_fit):
        assert ped_fit["model"].opti_REML.converged is True

    @pytest.mark.parametrize("i, key", [(0, "var_a"), (1, "var_d"), (2, "var_r")])
    def test_variances(self, ped_fit, i, key):
        actual = float(torch.exp(ped_fit["var"][i]))
        np.testing.assert_allclose(actual, EXPECTED_PED[key], rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("block, key", [(0, "blup_a"), (1, "blup_d")])
    def test_blup(self, ped_fit, block, key):
        L = ped_fit["L"]
        u = _np(ped_fit["model"].uhat).ravel()[block * L:(block + 1) * L]
        actual, expected = _ped_aligned(u, ped_fit["index"], key)
        np.testing.assert_allclose(actual, expected, atol=5e-4)

    def test_pev_a(self, ped_fit):
        L = ped_fit["L"]
        pev = np.diag(_np(ped_fit["model"].PEV))[:L]
        actual, se = _ped_aligned(pev, ped_fit["index"], "se_a")
        np.testing.assert_allclose(actual, se ** 2, rtol=2e-3)


# --------------------------------------------------------------------------- #
# Spatial: supplied Euclidean distance, exp(-rho * D)
# --------------------------------------------------------------------------- #
EXPECTED_SPAT = _load("spat_dist.json")


def _pairwise_dist(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


@pytest.fixture(scope="module", params=PATHS)
def spat_fit(request):
    # MUST match SUBSET_BLOCS / _grid in test_spatial.py
    data = DF.loc[(DF["year"] == 2000) & DF["BLOC"].isin(["B3", "B13"])].copy()
    data["ID"] = np.arange(len(data))
    dist = _pairwise_dist(data[["X", "Y"]].to_numpy(dtype=float))

    design = _dense_design(
        data=data,
        response="height",
        fixed="1",
        random=Random(unit="ID", right_hand="dist", distance=dist,
                      matrix_index=data["ID"].tolist()),
    )
    index = design.random[0].index.tolist()
    L = len(index)

    pos = {u: i for i, u in enumerate(data["ID"].tolist())}
    idx = [pos[u] for u in index]
    D_t = torch.as_tensor(dist[np.ix_(idx, idx)], dtype=torch.double, device=DEVICE)

    log_v0 = float(np.log(design.y.var().item()))
    log_su, log_se, log_rho = _leaf(log_v0), _leaf(log_v0), _leaf(0.0)

    def kernel(self):
        return torch.exp(-torch.exp(log_rho) * D_t.to(self.dtype))

    def varmeth(self):
        G = torch.exp(log_su) * kernel(self)
        R = torch.exp(log_se) * torch.eye(self.n, dtype=self.dtype, device=self.device)
        return G, R

    def varmeth_inv(self):
        Lk = torch.linalg.cholesky(kernel(self))
        Ginv = torch.exp(-log_su) * torch.cholesky_inverse(Lk)
        logdet_G = L * log_su + 2.0 * torch.log(torch.diagonal(Lk)).sum()
        Rinv = torch.exp(-log_se) * torch.eye(self.n, dtype=self.dtype, device=self.device)
        logdet_R = self.n * log_se
        return Ginv, Rinv, logdet_G, logdet_R

    model = _fit_low_level(design, [log_su, log_se, log_rho], varmeth, varmeth_inv, request.param)

    # level order -> data row order, as in the reference
    to_rows = [index.index(u) for u in data["ID"].tolist()]
    return dict(model=model, path=request.param, to_rows=to_rows,
                log_su=log_su, log_se=log_se, log_rho=log_rho)


class TestLowLevelSpatial:

    def test_routing(self, spat_fit):
        _assert_routing(spat_fit)

    def test_convergence(self, spat_fit):
        assert spat_fit["model"].opti_REML.converged is True

    def test_rho(self, spat_fit):
        actual = float(torch.exp(spat_fit["log_rho"]))
        np.testing.assert_allclose(actual, EXPECTED_SPAT["rho"][0], atol=1e-3)

    def test_Vu(self, spat_fit):
        actual = float(torch.exp(spat_fit["log_su"]))
        np.testing.assert_allclose(actual, EXPECTED_SPAT["Vu"], rtol=1e-3)

    def test_Ve(self, spat_fit):
        actual = float(torch.exp(spat_fit["log_se"]))
        np.testing.assert_allclose(actual, EXPECTED_SPAT["Ve"], rtol=1e-3)

    def test_intercept(self, spat_fit):
        actual = _np(spat_fit["model"].beta).ravel()[0]
        np.testing.assert_allclose(actual, EXPECTED_SPAT["beta"], rtol=1e-3)

    def test_eev_intercept(self, spat_fit):
        actual = spat_fit["model"].EEV.item()
        np.testing.assert_allclose(actual, EXPECTED_SPAT["eev_intercept"], rtol=2e-4)

    def test_blup(self, spat_fit):
        u = _np(spat_fit["model"].uhat).ravel()[spat_fit["to_rows"]]
        np.testing.assert_allclose(u, EXPECTED_SPAT["blup"], rtol=1e-3)

    def test_pev_diag(self, spat_fit):
        pev = np.diag(_np(spat_fit["model"].PEV))[spat_fit["to_rows"]]
        np.testing.assert_allclose(pev, EXPECTED_SPAT["pev_diag"], rtol=2e-3)


# --------------------------------------------------------------------------- #
# Random regression: intercept + slope per BLOC, unstructured S (lme4)
# --------------------------------------------------------------------------- #
EXPECTED_REG = _load("regression_random.json")


@pytest.fixture(scope="module", params=PATHS)
def reg_fit(request):
    df = DF[DF["year"] == 2000].copy()
    for col in ("circumference", "height"):
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    df["ID"] = np.arange(len(df))

    design = _dense_design(
        data=df,
        response="height",
        fixed="1 + circumference",
        random=Random(unit="BLOC", formula="1 + circumference", left_hand="full"),
    )
    rand = design.random[0]
    c, L = rand.c, rand.L

    # S = expm(A + A'), same parametrization as left_hand="full"
    A = _leaf(np.zeros((c, c)))
    log_se = _leaf(float(np.log(design.y.var().item())))

    # Z is component-outer / level-inner, hence G = S ⊗ I_L.
    # S is a dimensioned double tensor: it must be cast explicitly, otherwise
    # it promotes the float32 Z of the mixed-precision phase back to double.
    def varmeth(self):
        S = torch.linalg.matrix_exp(A + A.T).to(self.dtype)
        G = torch.kron(S, torch.eye(L, dtype=self.dtype, device=self.device))
        R = torch.exp(log_se) * torch.eye(self.n, dtype=self.dtype, device=self.device)
        return G, R

    def varmeth_inv(self):
        Sinv = torch.linalg.matrix_exp(-(A + A.T)).to(self.dtype)
        Ginv = torch.kron(Sinv, torch.eye(L, dtype=self.dtype, device=self.device))
        logdet_G = L * 2.0 * torch.trace(A)
        Rinv = torch.exp(-log_se) * torch.eye(self.n, dtype=self.dtype, device=self.device)
        logdet_R = self.n * log_se
        return Ginv, Rinv, logdet_G, logdet_R

    model = _fit_low_level(design, [A, log_se], varmeth, varmeth_inv, request.param)
    return dict(model=model, path=request.param, A=A, log_se=log_se, c=c, L=L)


class TestLowLevelRegression:

    def test_routing(self, reg_fit):
        _assert_routing(reg_fit)

    def test_convergence(self, reg_fit):
        assert reg_fit["model"].opti_REML.converged is True

    def test_beta(self, reg_fit):
        actual = _np(reg_fit["model"].beta).ravel()
        np.testing.assert_allclose(actual, EXPECTED_REG["beta"], atol=1e-4)

    def test_eev(self, reg_fit):
        actual = _np(reg_fit["model"].EEV)
        np.testing.assert_allclose(actual, EXPECTED_REG["eev"], atol=1e-4)

    def test_tvals(self, reg_fit):
        model = reg_fit["model"]
        actual = _np(model.beta).ravel() / np.sqrt(np.diag(_np(model.EEV)))
        np.testing.assert_allclose(actual, EXPECTED_REG["tvals"], rtol=1e-5)

    def test_varcorr(self, reg_fit):
        A = reg_fit["A"]
        actual = _np(torch.linalg.matrix_exp(A + A.T))
        np.testing.assert_allclose(actual, EXPECTED_REG["varcorr"], atol=1e-4)

    def test_sigma_r(self, reg_fit):
        actual = float(torch.exp(reg_fit["log_se"]))
        np.testing.assert_allclose(actual, EXPECTED_REG["sigma_r"], atol=1e-4)

    def test_blup(self, reg_fit):
        c, L = reg_fit["c"], reg_fit["L"]
        actual = _np(reg_fit["model"].uhat).reshape(c, L).T
        np.testing.assert_allclose(actual, EXPECTED_REG["blup"], atol=1e-4)

    def test_residuals(self, reg_fit):
        actual = _np(reg_fit["model"].residuals)
        np.testing.assert_allclose(actual, EXPECTED_REG["residuals"], atol=1e-4)

    def test_pev_diagonal_blocks(self, reg_fit):
        c, L = reg_fit["c"], reg_fit["L"]
        pev = _np(reg_fit["model"].PEV)
        expected = np.array(EXPECTED_REG["pev"])
        for i in range(L):
            idx = [i + j * L for j in range(c)]
            np.testing.assert_allclose(pev[np.ix_(idx, idx)], expected[:, :, i], atol=2.5e-3)