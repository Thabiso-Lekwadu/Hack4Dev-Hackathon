"""Experiment 2 — compare Stage-2 transfer strategies at n = 5, 25, 50, 100, 200 per class."""
import numpy as np, lightgbm as lgb, json, sys, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *
from spatial import smooth_proba

FSET = sys.argv[1] if len(sys.argv) > 1 else 'all'
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
if FSET.startswith('v2'):
    import pickle
    mad = pd.read_parquet(f'{CACHE}/features_madrid_v2.parquet'); ams = pd.read_parquet(f'{CACHE}/features_amsterdam_v2.parquet')
    G = pickle.load(open(f'{CACHE}/feature_groups_v2.pkl', 'rb'))
else:
    mad, ams, G = load()
core = G['baseline'] + G['anomaly'] + G['change'] + G['meta']
cols = {'all': core + G['spatial'], 'nospatial': core,
        'v2': core + G.get('spatial_v2', []), 'v2all': core + G['spatial'] + G.get('spatial_v2', [])}[FSET]
Xm, Xa = city_norm(mad[cols].values), city_norm(ams[cols].values)
ym, ya = mad.age_class.values, ams.age_class.values
pxa, pya = ams.px_key.values, ams.py_key.values

P = dict(n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
         class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1)
t0 = time.time()
M = lgb.LGBMClassifier(**P).fit(Xm, ym)
print(f'Madrid model fitted in {time.time()-t0:.0f}s', flush=True)
raw_a = M.predict(Xa, raw_score=True)           # (N,4) Madrid log-odds for Amsterdam
pa    = M.predict_proba(Xa)
lpa   = np.log(pa + 1e-6)
imp   = np.argsort(M.booster_.feature_importance('gain'))[::-1]
TOPK  = imp[:40]                                 # Madrid's 40 most useful features
print('zero-shot F1', round(macro_f1(ya, pa.argmax(1) + 1), 4), flush=True)

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def proto_proba(sup, Z):
    Zs = Z[sup]; c = np.unique(ya[sup])
    protos = np.stack([Zs[ya[sup] == k].mean(0) for k in c])
    d = np.sqrt(((Z[:, None, :] - protos[None]) ** 2).sum(-1))
    return softmax(-d / np.sqrt(Z.shape[1]) * 4)

def strategies(sup, n):
    out = {}
    out['zero'] = pa
    out['recal'] = LogisticRegression(C=1.0, max_iter=3000).fit(lpa[sup], ya[sup]).predict_proba(lpa)
    Z = Xa[:, TOPK]
    out['proto'] = proto_proba(sup, Z)
    S = np.hstack([lpa, Z]); sc = StandardScaler().fit(S[sup])
    C = 0.05 if n <= 5 else 0.1 if n <= 25 else 0.3
    out['stack'] = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(S[sup]), ya[sup]).predict_proba(sc.transform(S))
    # continued boosting: start from Madrid log-odds, add a few shallow trees fit on support
    rounds = int(min(20 + 1.5 * n, 300))
    bp = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=7,
              min_data_in_leaf=max(3, n // 5), feature_fraction=0.5, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=2)
    d = lgb.Dataset(Xa[sup], ya[sup] - 1, init_score=raw_a[sup].ravel(order='F'))
    b = lgb.train(bp, d, rounds)
    out['boost'] = softmax(raw_a + b.predict(Xa, raw_score=True))
    out['ens'] = (out['stack'] + out['boost'] + out['proto']) / 3
    return out

ALPHAS = [(0.0, 0), (0.5, 3), (0.7, 3), (0.7, 6)]
res = {}
for n in (5, 25, 50, 100, 200):
    scores = {}
    for sup, te in fewshot_splits(ya, n, 5, REPEATS):
        for name, pr in strategies(sup, n).items():
            for a, r in ALPHAS:
                p = smooth_proba(pxa, pya, pr, a, r) if a else pr
                scores.setdefault(f'{name}|a{a}r{r}', []).append(macro_f1(ya[te], p[te].argmax(1) + 1))
    res[n] = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in scores.items()}
    best = sorted(res[n].items(), key=lambda kv: -kv[1][0])[:6]
    print(n, best, f'{time.time()-t0:.0f}s', flush=True)
json.dump(res, open(f'{CACHE}/exp2_{FSET}.json', 'w'), indent=1)
