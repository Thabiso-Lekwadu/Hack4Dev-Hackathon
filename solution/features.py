"""
Step 2 — Per-pixel feature engineering on the (pixel x year x band) cube.

Every feature group below is motivated by Notebook 2's EDA or by the transfer goal.
Groups are kept separate so they can be ablated (switched on/off) in the experiments.
"""
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter

BASE_BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
INDEX_COLS = ['NDVI', 'NDBI', 'UI', 'MNDWI', 'BSI']
YEAR0, NYEARS = 1984, 42
YEARS = np.arange(YEAR0, YEAR0 + NYEARS)
# 5 roughly equal epochs. Finer than the baseline's 2 periods so the model can see
# *when* a change happened, which is exactly what separates classes 3 (1984-2004)
# and 4 (2004-2024).
EPOCHS = [(1984, 1991), (1992, 1999), (2000, 2007), (2008, 2015), (2016, 2025)]
EARLY_CUTOFF, LATE_START = 2003, 2004   # baseline split, kept for comparability


# ── Gap filling (vectorised version of Notebook 3's per-pixel interpolate) ─────
def gap_fill(cube):
    """Linear interpolation of interior gaps, nearest value at the edges."""
    P, T, B = cube.shape
    x   = cube.transpose(0, 2, 1).reshape(-1, T)          # (P*B, T)
    ok  = np.isfinite(x)
    idx = np.arange(T)[None, :].repeat(len(x), 0)
    prev_i = np.where(ok, idx, -1);  prev_i = np.maximum.accumulate(prev_i, axis=1)
    next_i = np.where(ok, idx, T);   next_i = np.minimum.accumulate(next_i[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(len(x))[:, None]
    pv = x[rows, np.clip(prev_i, 0, T - 1)]; nv = x[rows, np.clip(next_i, 0, T - 1)]
    pv = np.where(prev_i < 0, nv, pv);        nv = np.where(next_i >= T, pv, nv)
    span = (next_i - prev_i).astype(np.float32)
    w = np.where((prev_i >= 0) & (next_i < T) & (span > 0), (idx - prev_i) / np.maximum(span, 1), 0)
    out = np.where(ok, x, pv + w * (nv - pv)).astype(np.float32)
    return out.reshape(P, B, T).transpose(0, 2, 1)


def spectral_indices(c):
    """Same 5 normalised-difference indices as Notebook 3, applied per pixel-year."""
    B, G, R, N, S1, S2 = [c[..., i] for i in range(6)]
    e = 1e-6
    return np.stack([(N - R) / (N + R + e), (S1 - N) / (S1 + N + e), (S2 - N) / (S2 + N + e),
                     (G - S1) / (G + S1 + e), ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + e)], -1)


def year_anomaly(c):
    """
    DECISION — 'city-year anomaly' normalisation (the key transfer idea).
    For every year and band, subtract the city-wide median and divide by the city-wide
    IQR. This removes everything that is shared by all pixels of a city in a given year:
    sensor generation (L5/L7/L8/L9), atmosphere, sun angle, and the overall climate /
    material palette of the city (Mediterranean stucco vs Dutch brick). What is left is
    how a pixel differs from *its own city*, which is far more likely to mean the same
    thing in Madrid, Amsterdam, or Nairobi. Uses no labels, so it is legal for any new city.
    """
    med = np.nanmedian(c, axis=0, keepdims=True)
    q75, q25 = np.nanpercentile(c, [75, 25], axis=0, keepdims=True)
    return ((c - med) / (q75 - q25 + 1e-6)).astype(np.float32)


def stats_block(c, names, tag):
    """Mean / std / OLS slope over the full record + epoch means for a (P,T,K) series."""
    P, T, K = c.shape
    f = {}
    t = (YEARS - YEARS.mean()).astype(np.float32)
    mean = c.mean(1); std = c.std(1)
    slope = (c * t[None, :, None]).sum(1) / (t ** 2).sum()   # OLS slope per year
    resid = c - mean[:, None, :] - slope[:, None, :] * t[None, :, None]
    for k, n in enumerate(names):
        f[f'{n}_{tag}mean'] = mean[:, k]; f[f'{n}_{tag}std'] = std[:, k]
        f[f'{n}_{tag}slope'] = slope[:, k]; f[f'{n}_{tag}resstd'] = resid[:, :, k].std(1)
        for a, b in EPOCHS:
            m = (YEARS >= a) & (YEARS <= b)
            f[f'{n}_{tag}ep{a}'] = c[:, m, k].mean(1)
    return f


def changepoint(c):
    """
    DECISION — single change-point detection (construction-year proxy).
    Notebook 2 showed a break in the reflectance trajectory near the construction year
    for classes 3 and 4. We search every split year t and pick the one that maximises
    the between-segment variance  n1*n2/n * ||mean_before - mean_after||^2  (summed over
    standardised bands) — the classic binary-segmentation statistic. The returned
    break YEAR is directly informative: a break around 1990 -> class 3, around 2010 -> class 4,
    no strong break -> classes 1/2. Segments must be >= 3 years to avoid edge noise.
    """
    P, T, K = c.shape
    cs = np.cumsum(c, axis=1)
    tot = cs[:, -1, :]
    best = np.full(P, -1.0, np.float32); best_t = np.zeros(P, np.int16)
    best_diff = np.zeros((P, K), np.float32)
    for t in range(3, T - 3):
        n1, n2 = t, T - t
        m1 = cs[:, t - 1, :] / n1; m2 = (tot - cs[:, t - 1, :]) / n2
        d = m2 - m1
        score = (n1 * n2 / T) * (d ** 2).sum(1)
        upd = score > best
        best[upd] = score[upd]; best_t[upd] = t; best_diff[upd] = d[upd]
    # normalise the break strength by the residual noise so it is scale-free
    noise = np.median(np.abs(np.diff(c, axis=1)), axis=1).mean(1) + 1e-6
    out = {'cp_year': YEARS[best_t].astype(np.float32),
           'cp_strength': np.log1p(best / T) - np.log(noise),
           'cp_score_log': np.log1p(best)}
    return out, best_diff


def max_jump(c, names):
    """Largest single year-on-year change (on a 3-yr median-smoothed series) and its year."""
    from scipy.ndimage import median_filter
    sm = median_filter(c, size=(1, 3, 1), mode='nearest')
    d  = np.diff(sm, axis=1)
    f = {}
    for k, n in enumerate(names):
        i = np.abs(d[:, :, k]).argmax(1)
        f[f'{n}_jump'] = d[np.arange(len(d)), i, k]
        f[f'{n}_jump_year'] = YEARS[1:][i].astype(np.float32)
    return f


def spatial_context(px, py, feats, radii=(2, 6)):
    """
    DECISION — neighbourhood context (unlabelled spectral data only, NO labels).
    Buildings of the same era cluster in neighbourhoods (Notebook 4's maps show the RF
    already finds the right districts). So for a few key features we add the mean over
    building pixels within a (2r+1)x(2r+1) window: r=2 (150 m) ~ a block, r=6 (390 m)
    ~ a neighbourhood. Also the local building-pixel density (urban morphology).
    Only spectra are averaged, never labels, so there is no label leakage.
    """
    x = (px - px.min()).astype(int); y = (py - py.min()).astype(int)
    H, W = y.max() + 1, x.max() + 1
    occ = np.zeros((H, W), np.float32); occ[y, x] = 1
    out = {}
    for r in radii:
        s = 2 * r + 1
        cnt = uniform_filter(occ, s, mode='constant') * s * s
        out[f'nb{r}_density'] = (cnt[y, x] / (s * s)).astype(np.float32)
        for name, v in feats.items():
            g = np.zeros((H, W), np.float32); g[y, x] = v
            sm = uniform_filter(g, s, mode='constant') * s * s
            out[f'nb{r}_{name}'] = (sm[y, x] / np.maximum(cnt[y, x], 1)).astype(np.float32)
    return out


def build_features(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    cube, nobs = z['cube'], z['nobs']
    meta = pd.DataFrame({'px_key': z['px'], 'py_key': z['py'],
                         'weighted_mean_year': z['wmy'], 'age_class': z['y'].astype(int),
                         'city': str(z['city'])})
    has_early = (nobs[:, YEARS <= EARLY_CUTOFF] > 0).any(1).astype(np.float32)
    has_late  = (nobs[:, YEARS >= LATE_START] > 0).any(1).astype(np.float32)
    filled = gap_fill(cube)
    idx    = spectral_indices(filled)
    F = {}

    # ── (A) Baseline feature set — exact re-implementation of Notebook 3 (60 features)
    full = np.concatenate([filled, idx], -1); nm = BASE_BANDS + INDEX_COLS
    for k, n in enumerate(nm):
        F[f'{n}_mean'] = full[:, :, k].mean(1); F[f'{n}_std'] = full[:, :, k].std(1, ddof=1)
    e = YEARS <= EARLY_CUTOFF; l = YEARS >= LATE_START
    for k, n in enumerate(BASE_BANDS):
        F[f'{n}_early_mean'] = filled[:, e, k].mean(1); F[f'{n}_early_std'] = filled[:, e, k].std(1, ddof=1)
        F[f'{n}_late_mean']  = filled[:, l, k].mean(1); F[f'{n}_late_std']  = filled[:, l, k].std(1, ddof=1)
    d = np.diff(filled, axis=1)
    for k, n in enumerate(BASE_BANDS):
        F[f'd_{n}_mean'] = d[:, :, k].mean(1); F[f'd_{n}_std'] = d[:, :, k].std(1, ddof=1)
    F['has_early_data'] = has_early; F['has_late_data'] = has_late
    groups = {'baseline': list(F.keys())}

    # ── (B) City-year anomaly series: bands + indices, full stats + epochs
    anom = year_anomaly(np.concatenate([filled, idx], -1))
    G = stats_block(anom, [f'a{n}' for n in nm], '')
    F.update(G); groups['anomaly'] = list(G.keys())

    # ── (C) Change-point + max-jump on the anomaly series
    cp, cpd = changepoint(anom[:, :, :6])
    G = dict(cp)
    for k, n in enumerate(BASE_BANDS): G[f'cp_delta_{n}'] = cpd[:, k]
    G.update(max_jump(anom[:, :, [0, 3, 4, 6]], ['aBlue', 'aNIR', 'aSWIR1', 'aNDVI']))
    F.update(G); groups['change'] = list(G.keys())

    # ── (D) Observation quality / coverage
    G = {'coverage': z['coverage'].astype(np.float32),
         'n_obs_years': (nobs > 0).sum(1).astype(np.float32),
         'mean_scenes_per_year': nobs.mean(1).astype(np.float32)}
    F.update(G); groups['meta'] = list(G.keys())

    # ── (E) Spatial context on a compact set of transferable anomaly features
    key = {'aNIR_mean': F['aNIR_mean'], 'aSWIR1_mean': F['aSWIR1_mean'], 'aNDVI_mean': F['aNDVI_mean'],
           'aBlue_mean': F['aBlue_mean'], 'aNDBI_mean': F['aNDBI_mean'], 'cp_year': F['cp_year'],
           'cp_strength': F['cp_strength'], 'aNIR_slope': F['aNIR_slope'], 'coverage': G['coverage']}
    G = spatial_context(meta['px_key'].values, meta['py_key'].values, key)
    F.update(G); groups['spatial'] = list(G.keys())

    X = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
    return pd.concat([meta, X], axis=1), groups


if __name__ == '__main__':
    import pickle, sys
    out = {}
    for city in ['madrid', 'amsterdam']:
        df, groups = build_features(f'../data/cache/cube_{city}.npz')
        print(city, df.shape, {g: len(c) for g, c in groups.items()}, 'NaN:', int(df.isna().sum().sum()), flush=True)
        df.to_parquet(f'../data/cache/features_{city}.parquet')
        out['groups'] = groups
    pickle.dump(out['groups'], open('../data/cache/feature_groups.pkl', 'wb'))
