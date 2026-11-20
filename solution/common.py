"""Shared helpers: data loading, city normalisation, few-shot evaluation protocol."""
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
CACHE = '../data/cache'
META = ['px_key', 'py_key', 'weighted_mean_year', 'age_class', 'city']

def load():
    mad = pd.read_parquet(f'{CACHE}/features_madrid.parquet')
    ams = pd.read_parquet(f'{CACHE}/features_amsterdam.parquet')
    groups = pickle.load(open(f'{CACHE}/feature_groups.pkl', 'rb'))
    return mad, ams, groups

def city_norm(X):
    """
    DECISION — per-city robust standardisation (median / IQR of the city's own pixels).
    Fit separately on each city using ALL its pixels (labels are not used), so a feature
    value means "how unusual is this pixel for its city". Equivalent in spirit to
    AdaBN / per-domain batch-norm in deep domain adaptation. Robust stats are used
    because reflectance has heavy tails (haze, saturated roofs).
    """
    X = np.asarray(X, np.float32)
    med = np.median(X, 0); iqr = np.percentile(X, 75, 0) - np.percentile(X, 25, 0)
    iqr = np.where(iqr < 1e-6, X.std(0) + 1e-6, iqr)
    return np.clip((X - med) / iqr, -10, 10).astype(np.float32)

def fewshot_splits(y, n, n_folds=5, n_repeats=2, seed=SEED):
    """
    Evaluation protocol for the Amsterdam few-shot models (challenge step 4).
    Repeated stratified K-fold: the held-out fold is the test set; the labelled support
    set is n pixels per class sampled (fixed seed) from the remaining folds.
    Yields (support_idx, test_idx). n_folds*n_repeats scores per n -> mean +/- std.
    """
    rng = np.random.default_rng(seed)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(np.zeros(len(y)), y):
            sup = np.concatenate([rng.choice(tr[y[tr] == c], n, replace=False) for c in np.unique(y)])
            yield sup, te

def macro_f1(y, p):
    return f1_score(y, p, average='macro')
