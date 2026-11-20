"""
Spatial context v2 — extends features.py group (E).

DECISION — the ablation (exp1) showed spatial context is the single biggest gain
(Madrid CV 0.680 -> 0.729, 25-shot 0.591 -> 0.618). v2 adds:
  * a third, wider radius r=12 (~750 m: district scale, e.g. a whole post-war estate),
  * more neighbourhood variables: epoch means (WHEN the neighbourhood changed), slopes,
    break-strength and the NIR/SWIR change at the break,
  * local contrast = pixel value minus its neighbourhood mean (is this pixel unusual
    for its block? e.g. an infill building in an old street).
Still only spectral features of neighbours — never labels.
"""
import numpy as np, pandas as pd
from features import spatial_context

V2_VARS = ['aNIR_mean', 'aSWIR1_mean', 'aSWIR2_mean', 'aNDVI_mean', 'aBlue_mean', 'aRed_mean',
           'aNDBI_mean', 'aBSI_mean', 'cp_year', 'cp_strength', 'cp_delta_NIR', 'cp_delta_SWIR1',
           'aNIR_slope', 'aNDVI_slope', 'aBlue_slope', 'aNIR_ep1984', 'aNIR_ep2000', 'aNIR_ep2016',
           'aNDVI_ep1984', 'aNDVI_ep2000', 'aNDVI_ep2016', 'coverage', 'aNIR_std', 'aBlue_std']
CONTRAST_VARS = ['aNIR_mean', 'aNDVI_mean', 'cp_year', 'aBlue_mean', 'coverage']


def add_spatial_v2(df, radii=(2, 6, 12)):
    feats = {v: df[v].values.astype(np.float32) for v in V2_VARS}
    sp = spatial_context(df.px_key.values, df.py_key.values, feats, radii)
    sp = {f'v2_{k}': v for k, v in sp.items()}
    for v in CONTRAST_VARS:
        for r in (2, 12):
            sp[f'v2_ctr{r}_{v}'] = feats[v] - sp[f'v2_nb{r}_{v}']
    return pd.DataFrame(sp, index=df.index)
