"""
Freeze the reference for the factor-analytic multi-environment test.

Simulates a multi-environment trial and records what WOMBAT makes of it. The
test that follows compares pyreml's FA against WOMBAT's, so this script must
stay independent of pyreml's FA implementation -- that is the thing under
scrutiny. Only A_genomic is borrowed from pyreml, and only to build K.

THE DESIGN. N_ENV environments are N_ENV traits: every line carries its own
breeding value in each environment, correlated across environments through the
genetic covariance, and independent residuals since two environments are two
separate trials. So the model is

    u ~ MN(0, Sigma_A (x) K),   e ~ N(0, diag(sigma_e^2) (x) I)

with Sigma_A full rank and its correlation matrix drawn from a flat Wishart:
W ~ Wishart(I_q, df = q + 1) rescaled to a correlation matrix is exactly the
UNIFORM distribution over correlation matrices (LKJ eta = 1) -- any other df
would concentrate towards the identity (larger) or the boundary (smaller).

WHAT IS AND IS NOT TESTED. Sigma_A is full rank, so an FA with N_AXES < q is a
reduced-rank APPROXIMATION of it. The truth is therefore not the criterion:
neither solver can recover a rank-20 matrix with 2 factors, and neither is
expected to. The criterion is that two independent implementations, started
from the same point, land on the same approximation. The simulated Sigma_A is
stored all the same, as context for reading a disagreement.

THE STARTING POINT is written out explicitly rather than read off either
solver, so that the reference does not smuggle pyreml's initialization into
WOMBAT's run. SIGMA0 is decomposed here, in numpy, into the rank-N_AXES common
part and the specific diagonal, following the same eigen convention pyreml's
`left_hand="fa"` uses on its `init` argument -- so passing SIGMA0 to pyreml and
the derived pair to WOMBAT starts both at the same place.

Assumes the wombat binary sits in the current directory. Run from
tests/reference/. Writes ../data/arabido_fa.json and ../data/arabido_fa.npz.
"""

# %% load packages
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import wishart

from pyreml import A_genomic

# %% config
SEED = 42

GENO = "../data/arabido.npz"
OUT_JSON = "../data/arabido_fa.json"
OUT_NPZ = "../data/arabido_fa.npz"

N_ENV = 5                  # environments = traits
N_AXES = 2                  # factors the FA is fitted with

# Lines are subsampled because WOMBAT solves the dense MME: N_IND * N_ENV
# equations, so the cost grows as (N_IND * N_ENV)^3. At the full panel and 20
# environments that is 22500 equations and the run is measured in hours. Set to
# None to keep every line, but expect the reference to take a while.
N_IND = 200

SIGMA_G = 1.0               # genetic variance, common to every environment
RES_LOGRATIO_SD = 0.4       # spread of the per-environment residual variances
ENV_MEAN_SD = 2.0           # spread of the environment means

# Common starting point imposed on both solvers. Mild off-diagonal so that the
# leading factors are not degenerate at iteration zero.
SIGMA0 = 0.9 * np.eye(N_ENV) + 0.1 * np.ones((N_ENV, N_ENV))
RES0 = np.ones(N_ENV)

WOMBAT_EXE = Path("wombat").resolve()
WOMBAT_ALGO = "--pxai"
WORK = Path("wombat_arabido_fa")


# %% genotypes and kinship
with np.load(GENO) as z:
    # A_genomic writes into its argument (it overwrites monomorphic columns in
    # place), so this cast must stay a fresh array
    snp = z["snp"].astype(np.float64) - 1.0
    individuals = z["individual"]

if N_IND is not None and N_IND < len(individuals):
    # deterministic prefix rather than a draw: the artefact's line order is
    # already arbitrary, and a prefix keeps the subset reproducible without a
    # second stream of random numbers
    snp = snp[:N_IND]
    individuals = individuals[:N_IND]

K = A_genomic(snp, min_MAF=0.05, max_missing=0.1, shrink=True)
del snp

n = len(individuals)
print(f"{n} lines, {N_ENV} environments, {n * N_ENV} records")

Kinv = np.linalg.inv(K)
Klogdet = float(np.linalg.slogdet(K)[1])


# %% simulate
rng = np.random.default_rng(np.random.SeedSequence([SEED, n, N_ENV]))

# Flat prior over correlation matrices: a Wishart with identity scale and
# df = q + 1, rescaled to unit diagonal, is uniform over the space. The
# genetic variances are held equal across environments, so the whole structure
# the FA has to capture lives in the correlations.
W = wishart.rvs(df=N_ENV + 1, scale=np.eye(N_ENV), random_state=rng)
w_sd = np.sqrt(np.diag(W))
corr_A = W / np.outer(w_sd, w_sd)
np.fill_diagonal(corr_A, 1.0)
Sigma_A = SIGMA_G * corr_A

res_var = np.exp(rng.normal(scale=RES_LOGRATIO_SD, size=N_ENV))
env_mean = rng.normal(scale=ENV_MEAN_SD, size=N_ENV)

# u ~ MN(0, Sigma_A (x) K): a standard normal (n, q) sheet pre-multiplied by
# chol(K) and post-multiplied by chol(Sigma_A)'
Lk = np.linalg.cholesky(K)
La = np.linalg.cholesky(Sigma_A)
u = Lk @ rng.standard_normal((n, N_ENV)) @ La.T
del Lk, La

y = env_mean[None, :] + u + rng.standard_normal((n, N_ENV)) * np.sqrt(res_var)[None, :]

env_labels = [f"env_{e:02d}" for e in range(N_ENV)]


# %% starting point, decomposed once for both solvers
# Same convention as pyreml's left_hand="fa" applied to `init`: the N_AXES
# dominant eigenpairs carry the common part, and the specific diagonal takes
# whatever the low-rank part leaves on the diagonal. Reproduced here in numpy
# so the reference owes nothing to the implementation it will be used to check.
eigval, eigvec = np.linalg.eigh(SIGMA0)
Lambda0 = eigval[::-1][:N_AXES]
Q0 = eigvec[:, ::-1][:, :N_AXES]
FF0 = (Q0 * Lambda0) @ Q0.T                      # q x q, rank N_AXES
Psi0 = np.diag(SIGMA0) - np.diag(FF0)


# %% WOMBAT input
def write_gin(path, Kinv, Klogdet):
    """Kinv as a lower triangle in WOMBAT's (column, row, value) order,
    preceded by logdet(K)."""
    il = np.tril_indices(Kinv.shape[0])            # il[0] = row >= il[1] = column
    trip = np.column_stack([il[1] + 1, il[0] + 1, Kinv[il]])
    with open(path, "w") as f:
        f.write(f"{Klogdet:.10f}\n")
        np.savetxt(f, trip, fmt=["%d", "%d", "%.10e"])


def tri_vals(M):
    """Upper-triangle values, row-major, one per line. One value per line side-
    steps WOMBAT's record-length limit, which 20 traits would otherwise hit."""
    q = M.shape[0]
    return "\n".join(f"{M[i, j]:.10e}" for i in range(q) for j in range(i, q))


if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

# long format: traitno, animal, bnimal, mu, y. The code column is written twice
# because WOMBAT forbids one column serving two effects, and the FA idiom needs
# the same subject under two random effects.
code = np.arange(1, n + 1)
long = np.column_stack([
    np.repeat(np.arange(1, N_ENV + 1), n),         # traitno
    np.tile(code, N_ENV),                          # animal
    np.tile(code, N_ENV),                          # bnimal
    np.ones(n * N_ENV, dtype=int),                 # mu
    y.T.reshape(-1),                               # y, trait-outer
])
long = long[np.lexsort((long[:, 0], long[:, 1]))]  # by animal, then trait
np.savetxt(WORK / "data.dat", long, fmt=["%d", "%d", "%d", "%d", "%.10e"])

# two identical .gin: the same K carries the common and the specific effect
write_gin(WORK / "animal.gin", Kinv, Klogdet)
write_gin(WORK / "bnimal.gin", Kinv, Klogdet)

# `animal` (common, rank N_AXES) and `bnimal` (specific, DIAG) differ only by
# their first letter, which is how WOMBAT recognizes an FA. The residual is
# DIAG: two environments are two trials, so there is no residual covariance.
par = f"""COMMENT arabidopsis MET, q={N_ENV} axes={N_AXES}
ANALYSIS MUV PC {N_ENV}
DATA data.dat GRP
TRNOS {" ".join(str(i) for i in range(1, N_ENV + 1))}
traitno {N_ENV}
animal {n}
bnimal {n}
mu 1
NAMES {" ".join(env_labels)}
END
MODEL
SUBJ animal
FIX mu
RAN animal gin
RAN bnimal gin
{chr(10).join(f"tr {lab} {i + 1}" for i, lab in enumerate(env_labels))}
END MODEL
VAR animal {N_ENV} {N_AXES}
{tri_vals(FF0)}
VAR bnimal {N_ENV} {N_ENV} DIAG
{chr(10).join(f"{v:.10e}" for v in Psi0)}
VAR residual {N_ENV} {N_ENV} DIAG
{chr(10).join(f"{v:.10e}" for v in RES0)}
"""
(WORK / "wombat.par").write_text(par)


# %% WOMBAT run
env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
cmd = [str(WOMBAT_EXE), "-t", "--batch", "--dense", "--nohead",
       WOMBAT_ALGO, "wombat.par"]

print(f"running wombat ({WOMBAT_ALGO}) ...")
t0 = time.time()
proc = subprocess.run(cmd, cwd=WORK, env=env, capture_output=True, text=True)
elapsed = time.time() - t0

# WOMBAT exits 0 even on a .par error, so the output file is the real verdict
if not (WORK / "SumEstimates.out").exists():
    log = (WORK / "WOMBAT.log").read_text() if (WORK / "WOMBAT.log").exists() else "(no log)"
    raise RuntimeError(
        f"wombat produced no SumEstimates.out (rc={proc.returncode})\n"
        f"--- WOMBAT.log ---\n{log}\n--- stderr ---\n{proc.stderr}"
    )
print(f"  {elapsed:.1f} s")


# %% WOMBAT output
def floats(s):
    """Every parseable float on a line; labels such as 'Value' are dropped."""
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def re_block(txt, name):
    """Lines of the '***** Estimates for RE ... "name"' block, up to the next
    '*****'. The quotes in the pattern keep "animal" from matching "a+bnimal"."""
    lines = txt.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if "Estimates for RE" in l and f'"{name}"' in l), None)
    if start is None:
        raise RuntimeError(f'RE block "{name}" not found in SumEstimates.out')
    end = next((i for i in range(start + 1, len(lines)) if "*****" in lines[i]),
               len(lines))
    return lines[start:end]


def read_eigen(block, k):
    """WOMBAT's printed eigen-decomposition of a reduced-rank block.

    The 'Value' line wraps past six columns, so continuation lines are consumed
    until the '(%)' line; eigenvector rows are one trait per line, prefixed by
    the row index, and at 20 traits they do NOT wrap because the table is k
    columns wide, not q.
    """
    vi = next(i for i, l in enumerate(block)
              if l.strip().lower().startswith("eigenvalues of covariance"))
    vals = []
    for l in block[vi + 1:]:
        if l.strip().startswith("(%)"):
            break
        vals += floats(l)
    Lam = np.array(vals[:k], dtype=float)

    ei = next(i for i, l in enumerate(block)
              if l.strip().lower().startswith("eigenvectors of covariance"))
    rows = []
    for l in block[ei + 1:]:
        f = floats(l)
        if len(f) < 1 + k:                        # left the eigenvector table
            break
        rows.append(f[1:1 + k])                   # f[0] = row index
    return Lam, np.array(rows, dtype=float)


def read_variance_list(block, q, header="variance components"):
    """A DIAG effect's variances, printed as an index + value list rather than
    as a matrix."""
    vi = next((i for i, l in enumerate(block)
               if l.strip().lower().startswith(header)), None)
    if vi is None:
        raise RuntimeError(f"'{header}' not found in the block")
    vals = []
    for l in block[vi + 1:]:
        f = floats(l)
        if len(f) < 2:
            break
        vals.append(f[1])                         # f[0] = row index
        if len(vals) == q:
            break
    return np.array(vals, dtype=float)


def read_residual(txt, q):
    """Residual variances.

    Declared DIAG, so WOMBAT is expected to print a 'Variance components' list.
    Older versions print the full covariance matrix regardless, hence the
    fallback: the lower triangle is preceded by its row index, so the flat
    stream reads [1, a11, 2, a21, a22, ...] and consuming it positionally is
    wrap-agnostic.
    """
    lines = txt.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if "residual covariance" in l.lower()
                  or "Estimates of residual" in l), None)
    if start is None:
        raise RuntimeError("residual block not found in SumEstimates.out")
    block = lines[start:]

    vi = next((i for i, l in enumerate(block[:200])
               if l.strip().lower().startswith("variance components")), None)
    if vi is not None:
        return read_variance_list(block, q)

    ci = next(i for i, l in enumerate(block)
              if l.strip().lower().startswith("covariance matrix"))
    vals = []
    for l in block[ci + 1:]:
        if l.strip().lower().startswith("eigenvalues"):
            break
        vals += floats(l)
    R = np.zeros((q, q))
    pos = 0
    for i in range(q):
        if pos >= len(vals) or int(round(vals[pos])) != i + 1:
            raise RuntimeError("unexpected layout in the residual block")
        pos += 1
        for j in range(i + 1):
            R[i, j] = R[j, i] = vals[pos]
            pos += 1
    return np.diag(R).copy()


def read_intercepts(work, q):
    """The q intercepts from FixSolutions.out, in trait order; each 'mu' line
    carries its solution as the last float."""
    vals = []
    for l in (work / "FixSolutions.out").read_text().splitlines():
        if "mu" in l.split():
            f = floats(l)
            if f:
                vals.append(f[-1])
    if len(vals) != q:
        raise RuntimeError(f"expected {q} intercepts, found {len(vals)}")
    return np.array(vals, dtype=float)


txt = (WORK / "SumEstimates.out").read_text()
conv_txt = txt + ((WORK / "WOMBAT.log").read_text()
                  if (WORK / "WOMBAT.log").exists() else "")
converged = "convergence has not been achieved" not in conv_txt.lower()

Lam_w, Q_w = read_eigen(re_block(txt, "animal"), N_AXES)
Psi_w = read_variance_list(re_block(txt, "bnimal"), N_ENV)
res_w = read_residual(txt, N_ENV)
mu_w = read_intercepts(WORK, N_ENV)

# the fitted genetic covariance, reassembled from what WOMBAT printed:
# Gamma Gamma' + diag(Psi) with Gamma = Q sqrt(Lambda). This reproduces the
# 'a+bnimal' block, which is therefore not parsed separately.
Gamma_w = Q_w * np.sqrt(Lam_w)
Sigma_A_w = Gamma_w @ Gamma_w.T + np.diag(Psi_w)


# %% freeze
reference = {
    "config": {
        "seed": SEED,
        "n_ind": n, "n_env": N_ENV, "n_axes": N_AXES,
        "sigma_g": SIGMA_G,
        "res_logratio_sd": RES_LOGRATIO_SD,
        "env_mean_sd": ENV_MEAN_SD,
        "env_labels": env_labels,
    },
    "start": {
        # SIGMA0 goes to pyreml as `init`; FF0 / Psi0 / RES0 are the same point
        # written in WOMBAT's idiom, derived here rather than read off pyreml
        "Sigma0": SIGMA0.tolist(),
        "FF0": FF0.tolist(),
        "Psi0": Psi0.tolist(),
        "Res0": RES0.tolist(),
    },
    "truth": {
        # context for reading a disagreement, NOT an assertion target: a
        # full-rank Sigma_A cannot be recovered by a rank-2 FA
        "Sigma_A": Sigma_A.tolist(),
        "corr_A": corr_A.tolist(),
        "res_var": res_var.tolist(),
        "env_mean": env_mean.tolist(),
    },
    "wombat": {
        "algo": WOMBAT_ALGO,
        "converged": bool(converged),
        "time": elapsed,
        "Lambda": Lam_w.tolist(),          # N_AXES, descending
        "Q": Q_w.tolist(),                 # (N_ENV, N_AXES), sign-arbitrary
        "Psi": Psi_w.tolist(),             # specific variances
        "Sigma_A": Sigma_A_w.tolist(),     # Gamma Gamma' + diag(Psi)
        "res_var": res_w.tolist(),
        "intercepts": mu_w.tolist(),
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(reference, f, indent=2)

np.savez_compressed(
    OUT_NPZ,
    y=y,                            # (n, N_ENV), wide
    u_true=u,                       # (n, N_ENV)
    individual=individuals,
)
