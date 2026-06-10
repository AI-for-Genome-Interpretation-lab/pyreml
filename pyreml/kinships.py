import numpy as np
import pandas as pd

def A_genomic(
        X,
        min_MAF: float = 0.05,
        max_missing: float = 0.1,
        shrink: bool = False
    ) :
    '''
    This is a direct retranscription of the A_mat
    function from the R paclage: rrBLUP
    X is a diploid species SNP matrix encoded: -1, 0, 1
    the following parameters are fixed:
        - impute.method = "mean"
        - shrink.method = "EJ"
    '''

    n = X.shape[0]

    frac_missing = np.isnan(X).mean(axis=0)
    freq = np.nanmean(X + 1, axis=0) / 2
    MAF = np.minimum(freq, 1 - freq)

    markers = np.where((MAF >= min_MAF) & (frac_missing <= max_missing))[0]
    m = len(markers)
    var_A = 2 * np.mean(freq[markers] * (1 - freq[markers]))
    mono = np.where(freq * (1 - freq) == 0)[0]
    X[:, mono] = 2 * freq[mono] - 1
    freq_mat = np.ones((n, 1)) @ freq[markers].reshape(1, -1)
    W = X[:, markers] + 1 - 2 * freq_mat

    # imputation
    W[np.isnan(W)] = 0

    if shrink:
        Z = W - W.mean(axis=1, keepdims=True)
        Z2 = Z ** 2
        S = (Z @ Z.T) / m
        target = np.mean(np.diag(S)) * np.eye(n)
        var_S = (Z2 @ Z2.T) / m**2 - S**2 / m
        b2 = var_S.sum()
        d2 = ((S - target) ** 2).sum()
        delta = float(np.clip(b2 / d2, 0, 1))
        print(f"Shrinkage intensity: {delta:.2f}")
        cov_W = target * delta + (1 - delta) * S
        W_mean = W.mean(axis=1)
        K = (cov_W + np.outer(W_mean, W_mean)) / var_A
    else:
        K = (W @ W.T) / var_A / m

    return K

def A_pedigree(pedigree: pd.DataFrame) -> np.ndarray:
    '''
    This is a direct retranscription of the makeD
    function from the R paclage: nadiv
    pedigree is a pandas dataframe with 3 columns:
        - ID: the individual
        - DAM and SIRE: its parents, may be NA
    '''
    ped = pedigree.copy()
    ped.columns = ped.columns.str.lower()

    # --- checks ---
    if not {"id", "dam", "sire"}.issubset(ped.columns):
        raise ValueError("Pedigree must have columns: id, dam, sire")

    if ped["id"].duplicated().any():
        dups = ped["id"][ped["id"].duplicated()].unique().tolist()
        raise ValueError(f"Duplicate individuals: {dups}")

    known_dam = ped["dam"].dropna().unique()
    known_sire = ped["sire"].dropna().unique()
    all_ids = set(ped["id"])
    missing_parents = (set(known_dam) | set(known_sire)) - all_ids
    if missing_parents:
        raise ValueError(f"Parents not in pedigree: {sorted(missing_parents)}")

    # --- generation assignment + sort ---
    ids = ped["id"].values
    dam = ped["dam"].values
    sire = ped["sire"].values
    n = len(ids)
    id_to_idx = {v: i for i, v in enumerate(ids)}

    gen = np.zeros(n, dtype=int)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            g = 0
            if pd.notna(dam[i]):
                g = max(g, gen[id_to_idx[dam[i]]] + 1)
            if pd.notna(sire[i]):
                g = max(g, gen[id_to_idx[sire[i]]] + 1)
            if g > gen[i]:
                gen[i] = g
                changed = True

    # sort by (gen, dam, sire) with NA first (like na.last=FALSE in R)
    dam_key = np.array([id_to_idx[d] + 1 if pd.notna(d) else 0 for d in dam])
    sire_key = np.array([id_to_idx[s] + 1 if pd.notna(s) else 0 for s in sire])
    sort_idx = np.lexsort((sire_key, dam_key, gen))

    ids_s = ids[sort_idx]
    dam_s = dam[sort_idx]
    sire_s = sire[sort_idx]

    # numeric pedigree (0-based indices in sorted order)
    id_to_sidx = {v: i for i, v in enumerate(ids_s)}
    dam_idx = np.array([id_to_sidx[d] if pd.notna(d) else n for d in dam_s])
    sire_idx = np.array([id_to_sidx[s] if pd.notna(s) else n for s in sire_s])

    # --- Meuwissen & Luo: f and dii ---
    f = np.zeros(n + 1)
    f[n] = -1.0
    dii = np.zeros(n)
    AN = np.empty(2 * n, dtype=float)
    li = np.zeros(n)

    for k in range(n):
        dii[k] = 0.5 - 0.25 * (f[dam_idx[k]] + f[sire_idx[k]])

        if k > 0 and dam_idx[k] == dam_idx[k-1] and sire_idx[k] == sire_idx[k-1]:
            f[k] = f[k-1]
        else:
            li[k] = 1.0
            ai = 0.0
            j = k
            cnt = 0
            while j >= 0:
                sj = sire_idx[j]
                dj = dam_idx[j]
                if sj < n:
                    AN[cnt] = sj
                    li[sj] += 0.5 * li[j]
                    cnt += 1
                if dj < n:
                    AN[cnt] = dj
                    li[dj] += 0.5 * li[j]
                    cnt += 1
                ai += li[j] * li[j] * dii[j]
                j = -n - 1  # empty sentinel
                for h in range(cnt):
                    if AN[h] > j:
                        j = int(AN[h])
                for h in range(cnt):
                    if AN[h] == j:
                        AN[h] = -n - 1
            f[k] = ai - 1.0
            li[:k+1] = 0.0

    # --- Tinv and assembly ---
    Tinv = np.eye(n)
    for i in range(n):
        if dam_idx[i] < n:
            Tinv[i, dam_idx[i]] = -0.5
        if sire_idx[i] < n:
            Tinv[i, sire_idx[i]] = -0.5

    sqrt_dii_inv = np.diag(np.sqrt(1.0 / dii))
    L = np.linalg.solve(sqrt_dii_inv @ Tinv, np.eye(n))
    A = L @ L.T

    # --- reorder to original pedigree order ---
    inv_sort = np.argsort(sort_idx)
    A = A[np.ix_(inv_sort, inv_sort)]

    return A

def D_pedigree(pedigree: pd.DataFrame) -> np.ndarray:
    '''
    This is a direct retranscription of the makeD
    function from the R paclage: nadiv
    pedigree is a pandas dataframe with 3 columns:
        - ID: the individual
        - DAM and SIRE: its parents, may be NA
    
    NB1: makeD is an approximation, it is only exact
    in the absence of inbreeding.

    NB2: whenever one or the other parent is missing,
    the individual is treated as a founder (both
    parents missing)
    '''
    A = A_pedigree(pedigree)

    ped = pedigree.copy()
    ped.columns = ped.columns.str.lower()

    ids = ped["id"].values
    dam = ped["dam"].values.copy()
    sire = ped["sire"].values.copy()

    # force founders when only one parent is known
    only_dam = pd.notna(dam) & pd.isna(sire)
    only_sire = pd.isna(dam) & pd.notna(sire)
    dam[only_dam] = None
    sire[only_sire] = None

    id_to_idx = {v: i for i, v in enumerate(ids)}
    n = len(ids)

    dam_idx = np.array([id_to_idx[d] if pd.notna(d) else -1 for d in dam])
    sire_idx = np.array([id_to_idx[s] if pd.notna(s) else -1 for s in sire])

    D = np.zeros((n, n))
    half_A = A / 2.0

    for k in range(n):
        if dam_idx[k] == -1:
            continue
        for j in range(k):
            if dam_idx[j] == -1:
                continue
            dk, sk = dam_idx[k], sire_idx[k]
            dj, sj = dam_idx[j], sire_idx[j]
            val = half_A[dk, dj] * half_A[sk, sj] + half_A[dk, sj] * half_A[sk, dj]
            if val != 0.0:
                D[k, j] = val
                D[j, k] = val

    np.fill_diagonal(D, 2.0 - np.diag(A))
    return D