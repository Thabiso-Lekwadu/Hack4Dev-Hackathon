"""
Data-leakage audit for the Hack4Dev building-age solution.

Leakage = information that would not be available when predicting a genuinely new
pixel/city sneaking into training or into the evaluation, so scores look better than
they will be in deployment. We check four kinds:

  1. Label leakage through features   : is any label (or a near-copy of it) a feature?
  2. Pipeline fit-scope leakage        : were scalers/statistics fitted on test data?
  3. Spatial leakage in evaluation     : random CV puts near-identical neighbours in
                                          train and test (spatial autocorrelation).
  4. Few-shot protocol leakage         : can Amsterdam test pixels reach the support set?

Outputs: data/cache/leakage_report.json + printed summary.
"""
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])  # repository root
import json, time, numpy as np, pandas as pd, lightgbm as lgb
from scipy.ndimage import maximum_filter
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

SEED = 42
FEAT = ROOT + '/data/features'
groups = json.load(open(f'{FEAT}/feature_groups.json'))
ALL = [c for g in ['baseline', 'anomaly', 'change', 'meta', 'spatial_v1', 'spatial'] for c in groups[g]]
CORE = [c for g in ['baseline', 'anomaly', 'change', 'meta'] for c in groups[g]]
mad = pd.read_parquet(f'{FEAT}/features_madrid.parquet')
ams = pd.read_parquet(f'{FEAT}/features_amsterdam.parquet')
ym, ya = mad.age_class.values.astype(int), ams.age_class.values.astype(int)
R = {}
f1 = lambda y, p: f1_score(y, p, average='macro')

def robust_stats(X):
    med = np.median(X, 0); iqr = np.percentile(X, 75, 0) - np.percentile(X, 25, 0)
    return med, np.where(iqr < 1e-6, X.std(0) + 1e-6, iqr)
def apply(X, st): return np.clip((X - st[0]) / st[1], -10, 10).astype(np.float32)
def city_std(X): X = np.asarray(X, np.float32); return apply(X, robust_stats(X))

# ── 1. Label leakage through features ─────────────────────────────────────────
META = {'px_key', 'py_key', 'weighted_mean_year', 'age_class', 'city'}
R['label_columns_in_features'] = sorted(META & set(ALL))
# single-feature power: a depth-3 tree on ONE feature, 3-fold CV. A leaked label shows up
# as one feature that alone scores close to the full model.
t0 = time.time()
Xm_all = city_std(mad[ALL].values)
rng = np.random.default_rng(SEED); sub = rng.choice(len(ym), 20000, replace=False)
single = {}
for j, c in enumerate(ALL):
    p = cross_val_predict(DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=SEED),
                          Xm_all[sub, j:j + 1], ym[sub], cv=3)
    single[c] = f1(ym[sub], p)
top = sorted(single.items(), key=lambda kv: -kv[1])[:8]
R['single_feature_top8'] = [(c, round(v, 4)) for c, v in top]
R['coverage_alone'] = round(single['coverage'], 4)
print('1. label columns in features:', R['label_columns_in_features'], '| best single feature:', top[0], f'[{time.time()-t0:.0f}s]', flush=True)

# ── 3 (+2). Madrid: random vs spatially blocked CV, with/without spatial features ─
P = dict(n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight='balanced',
         random_state=SEED, n_jobs=-1, verbose=-1)
px, py = mad.px_key.values, mad.py_key.values
BLOCK, BUFFER = 67, 12          # 67 px x 30 m ~ 2 km tiles; buffer = largest neighbourhood radius (12 px = 360 m)
tile = (px // BLOCK) * 100000 + (py // BLOCK)

def buffer_mask(test_idx):
    """True for pixels within BUFFER px (Chebyshev) of any test pixel."""
    x = px - px.min(); y = py - py.min()
    g = np.zeros((y.max() + 1, x.max() + 1), bool); g[y[test_idx], x[test_idx]] = True
    near = maximum_filter(g, size=2 * BUFFER + 1)
    return near[y, x]

def run(name, cols, split, train_only_scaling=False):
    X = mad[cols].values.astype(np.float32)
    if not train_only_scaling: X = city_std(X)
    scores, ntr = [], []
    for tr, te in split:
        if train_only_scaling:
            st = robust_stats(X[tr]); Xtr, Xte = apply(X[tr], st), apply(X[te], st)
        else:
            Xtr, Xte = X[tr], X[te]
        m = lgb.LGBMClassifier(**P).fit(Xtr, ym[tr])
        scores.append(f1(ym[te], m.predict(Xte))); ntr.append(len(tr))
    R[name] = dict(mean=round(float(np.mean(scores)), 4), std=round(float(np.std(scores)), 4),
                   folds=[round(s, 4) for s in scores], mean_train_size=int(np.mean(ntr)))
    print(name, R[name]['mean'], '+/-', R[name]['std'], f'train~{int(np.mean(ntr))}', f'[{time.time()-t0:.0f}s]', flush=True)

rand = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(ym)), ym))
blk = []
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(ym)), ym, tile):
    near = buffer_mask(te)
    blk.append((tr[~near[tr]], te))          # drop training pixels inside the buffer
R['block_setup'] = dict(block_px=BLOCK, block_km=round(BLOCK * 0.03, 2), buffer_px=BUFFER, buffer_m=BUFFER * 30,
                        n_tiles=int(len(np.unique(tile))))
run('madrid_random_all', ALL, rand)
run('madrid_random_trainonly_scaling', ALL, rand, train_only_scaling=True)
run('madrid_random_core', CORE, rand)
run('madrid_blocked_all', ALL, blk)
run('madrid_blocked_core', CORE, blk)
json.dump(R, open(ROOT + '/data/cache/leakage_report.json', 'w'), indent=1)

# ── 4. Amsterdam few-shot: random folds vs 2 km blocks with buffer ────────────
booster = lgb.Booster(model_file=ROOT + '/models/stage1_madrid_lgbm.txt')
Xa = city_std(ams[ALL].values)
raw = booster.predict(Xa, raw_score=True)
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
lp = np.log(softmax(raw) + 1e-6)
TOP = np.argsort(booster.feature_importance('gain'))[::-1][:40]
Z = Xa[:, TOP]
def adapt(sup, n):
    """Exact copy of Notebook 6's adapt_to_city (smoothing off, as selected on Madrid)."""
    bp = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=7, min_data_in_leaf=max(3, n // 5),
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=2)
    b = lgb.train(bp, lgb.Dataset(Xa[sup], ya[sup] - 1, init_score=raw[sup].ravel(order='F')), int(min(20 + 3 * n, 400)))
    p_boost = softmax(raw + b.predict(Xa, raw_score=True))
    S = np.hstack([lp, Z]); sc = StandardScaler().fit(S[sup])
    C = 0.05 if n <= 5 else 0.1 if n <= 25 else 0.3
    p_stack = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(S[sup]), ya[sup]).predict_proba(sc.transform(S))
    protos = np.stack([Z[sup][ya[sup] == c].mean(0) for c in (1, 2, 3, 4)])
    p_proto = softmax(-np.sqrt(((Z[:, None] - protos[None]) ** 2).sum(-1)) / np.sqrt(Z.shape[1]) * 4)
    w = max(0.0, 1 - n / 100)
    return (w * p_proto + p_stack + p_boost) / (w + 2)

apx, apy = ams.px_key.values, ams.py_key.values
atile = (apx // BLOCK) * 100000 + (apy // BLOCK)
def abuffer(test_idx):
    x = apx - apx.min(); y = apy - apy.min()
    g = np.zeros((y.max() + 1, x.max() + 1), bool); g[y[test_idx], x[test_idx]] = True
    return maximum_filter(g, size=2 * BUFFER + 1)[y, x]
overlap_violations = 0
for mode in ['random', 'blocked']:
    res = {}
    for n in (5, 25, 50, 100, 200):
        sc_ = []
        for rep in range(2):
            rng = np.random.default_rng(SEED + rep)
            if mode == 'random':
                splits = StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(np.zeros(len(ya)), ya)
            else:
                splits = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep).split(np.zeros(len(ya)), ya, atile)
            for tr, te in splits:
                if mode == 'blocked': tr = tr[~abuffer(te)[tr]]
                sup = np.concatenate([rng.choice(tr[ya[tr] == c], n, replace=False) for c in (1, 2, 3, 4)])
                overlap_violations += len(np.intersect1d(sup, te))
                p = adapt(sup, n)
                sc_.append(f1(ya[te], p[te].argmax(1) + 1))
        res[n] = (round(float(np.mean(sc_)), 4), round(float(np.std(sc_)), 4))
        print('amsterdam', mode, n, res[n], f'[{time.time()-t0:.0f}s]', flush=True)
    R[f'amsterdam_{mode}'] = res
R['support_test_overlap'] = int(overlap_violations)
json.dump(R, open(ROOT + '/data/cache/leakage_report.json', 'w'), indent=1)
print('DONE', flush=True)
