"""
Spatial post-processing of class probabilities.

DECISION — building age is spatially clustered (whole neighbourhoods are built in the same
decade; Notebook 4's maps show this clearly). A pixel's own spectrum is noisy, but the
*predicted* probabilities of its neighbours are extra evidence. We therefore blend each
pixel's probability vector with the mean prediction of the building pixels around it:

    p_smooth = (1 - alpha) * p_self + alpha * mean_{neighbours within r}(p)

Only model PREDICTIONS are averaged — never labels — so this works for an unlabelled city
and does not leak ground truth. alpha and r are tuned on Madrid only and then frozen.
"""
import numpy as np
from scipy.ndimage import uniform_filter


def smooth_proba(px, py, proba, alpha=0.5, r=3, use_mask=None):
    """
    px, py   : grid keys of the pixels being predicted
    proba    : (N, C) probabilities for those pixels
    use_mask : optional bool (N,) — only these pixels contribute as neighbours
               (used in Madrid CV so that only out-of-fold predictions are averaged).
    """
    if alpha <= 0:
        return proba
    x = (px - px.min()).astype(int); y = (py - py.min()).astype(int)
    H, W = y.max() + 1, x.max() + 1
    m = np.ones(len(px), bool) if use_mask is None else use_mask
    s = 2 * r + 1
    cnt_g = np.zeros((H, W), np.float32); cnt_g[y[m], x[m]] = 1
    cnt = uniform_filter(cnt_g, s, mode='constant')[y, x] * s * s
    out = np.empty_like(proba)
    for c in range(proba.shape[1]):
        g = np.zeros((H, W), np.float32); g[y[m], x[m]] = proba[m, c]
        tot = uniform_filter(g, s, mode='constant')[y, x] * s * s
        # exclude the pixel itself from its own neighbourhood mean
        self_in = m.astype(np.float32)
        nb = (tot - proba[:, c] * self_in) / np.maximum(cnt - self_in, 1)
        out[:, c] = nb
    has_nb = (cnt - m.astype(np.float32)) > 0
    sm = (1 - alpha) * proba + alpha * out
    sm[~has_nb] = proba[~has_nb]          # isolated pixels keep their own prediction
    return sm / sm.sum(1, keepdims=True)
