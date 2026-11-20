"""
Step 1 — Build a compact (pixel x year x band) reflectance cube for each city.

Design decisions (each one is deliberate):
  * Read only the columns we need, one parquet row-group at a time.
    -> The workspace has ~3 GB RAM; loading all 48 columns of Madrid (3.2M rows)
       with pandas would not fit comfortably.
  * Apply the haze filter (Blue > 15,000) PER OBSERVATION SLOT, not per row.
    -> The baseline drops a whole pixel-year when its first valid obs is hazy,
       even if obs_2 / obs_3 are clean. Filtering per slot keeps that data.
  * Combine the up-to-3 valid observations of a year with the MEDIAN.
    -> The baseline keeps only the first valid obs. The median of several clean
       summer scenes is a less noisy estimate of the surface reflectance.
  * Also drop observations with any band == 0 (corrupt rows found in Notebook 2).
  * Store float32 cubes: Madrid 76k px x 42 yr x 6 bands = ~77 MB.
"""
import sys, numpy as np, pyarrow.parquet as pq

BASE_BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
BLUE_MAX   = 15_000
YEAR0, NYEARS = 1984, 42           # 1984..2025 inclusive
CLASS_EVENTS = {'Amsterdam': 1945, 'Madrid': 1960}

def build(path, out):
    pf   = pq.ParquetFile(path)
    cols = ['city', 'year', 'px_key', 'py_key', 'coverage', 'weighted_mean_year']
    for s in (1, 2, 3):
        cols += [f'{b}_{s}' for b in BASE_BANDS] + [f'qa_valid_{s}']
    keys_all, chunks = [], []
    for rg in range(pf.num_row_groups):
        t   = pf.read_row_group(rg, columns=cols)
        d   = {c: t.column(c).to_numpy(zero_copy_only=False) for c in cols}
        n   = len(d['year'])
        obs = np.full((3, n, 6), np.nan, dtype=np.float32)
        for s in (1, 2, 3):
            v = np.stack([d[f'{b}_{s}'].astype(np.float32) for b in BASE_BANDS], 1)
            q = np.asarray(d[f'qa_valid_{s}'], dtype=object)
            q = np.array([bool(x) if x is not None else False for x in q])
            ok = q & np.isfinite(v).all(1) & (v[:, 0] <= BLUE_MAX) & (v > 0).all(1)
            obs[s - 1][ok] = v[ok]
        # median over the valid slots (nanmedian ignores the invalid ones)
        with np.errstate(all='ignore'):
            import warnings; warnings.simplefilter('ignore')
            med = np.nanmedian(obs, axis=0)
        nvalid = np.isfinite(obs[:, :, 0]).sum(0).astype(np.int8)
        keep = nvalid > 0
        chunks.append(dict(px=d['px_key'][keep].astype(np.int64), py=d['py_key'][keep].astype(np.int64),
                           yr=d['year'][keep].astype(np.int16), v=med[keep], nv=nvalid[keep],
                           cov=d['coverage'][keep].astype(np.float32),
                           wmy=d['weighted_mean_year'][keep].astype(np.float32),
                           city=str(d['city'][0])))
        del t, d, obs, med
        print(f'  row group {rg+1}/{pf.num_row_groups}: kept {keep.sum():,}/{n:,}', flush=True)
    px  = np.concatenate([c['px'] for c in chunks]); py = np.concatenate([c['py'] for c in chunks])
    yr  = np.concatenate([c['yr'] for c in chunks]); v  = np.concatenate([c['v'] for c in chunks])
    nv  = np.concatenate([c['nv'] for c in chunks]); cov = np.concatenate([c['cov'] for c in chunks])
    wmy = np.concatenate([c['wmy'] for c in chunks]); city = chunks[0]['city']
    key = px * 10_000_000 + py
    ukey, inv = np.unique(key, return_inverse=True)
    P = len(ukey)
    cube = np.full((P, NYEARS, 6), np.nan, dtype=np.float32)
    nobs = np.zeros((P, NYEARS), dtype=np.int8)
    cube[inv, yr - YEAR0] = v
    nobs[inv, yr - YEAR0] = nv
    # per-pixel constants (identical across years for a pixel)
    pix_cov = np.zeros(P, np.float32); pix_cov[inv] = cov
    pix_wmy = np.zeros(P, np.float32); pix_wmy[inv] = wmy
    ev   = CLASS_EVENTS[city]
    y    = np.digitize(pix_wmy, [ev, 1984, 2004], right=True) + 1   # same bins as pd.cut(right=True)
    np.savez_compressed(out, cube=cube, nobs=nobs, px=ukey // 10_000_000, py=ukey % 10_000_000,
                        coverage=pix_cov, wmy=pix_wmy, y=y.astype(np.int8), city=city)
    print(f'{city}: {P:,} pixels, cube {cube.shape}, obs-years filled {np.isfinite(cube[...,0]).mean():.1%}')
    print('  class counts:', {c: int((y == c).sum()) for c in (1, 2, 3, 4)})

if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
