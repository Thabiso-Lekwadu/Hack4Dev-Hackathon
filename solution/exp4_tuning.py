"""Experiment 4 — tune the Stage-2 ensemble: weights by n, boosting length, self-training."""
import numpy as np, lightgbm as lgb, json, sys, time, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *
from spatial import smooth_proba

FSET = sys.argv[1] if len(sys.argv) > 1 else 'v2all'
mad = pd.read_parquet(f'{CACHE}/features_madrid_v2.parquet'); ams = pd.read_parquet(f'{CACHE}/features_amsterdam_v2.parquet')
G = pickle.load(open(f'{CACHE}/feature_groups_v2.pkl', 'rb'))
core = G['baseline'] + G['anomaly'] + G['change'] + G['meta']
cols = {'v2': core + G['spatial_v2'], 'v2all': core + G['spatial'] + G['spatial_v2']}[FSET]
Xm, Xa = city_norm(mad[cols].values), city_norm(ams[cols].values)
ym, ya = mad.age_class.values, ams.age_class.values
pxa, pya = ams.px_key.values, ams.py_key.values
P = dict(n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
         class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1)
t0 = time.time()
M = lgb.LGBMClassifier(**P).fit(Xm, ym)
raw_a = M.predict(Xa, raw_score=True); pa = M.predict_proba(Xa); lpa = np.log(pa + 1e-6)
TOPK = np.argsort(M.booster_.feature_importance('gain'))[::-1][:40]
print(f'fit {time.time()-t0:.0f}s zero-shot', round(macro_f1(ya, pa.argmax(1) + 1), 4), flush=True)

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def proto(sup, Z, temp=4):
    c = np.unique(ya[sup]); pr = np.stack([Z[sup][ya[sup] == k].mean(0) for k in c])
    d = np.sqrt(((Z[:, None] - pr[None]) ** 2).sum(-1)); return softmax(-d / np.sqrt(Z.shape[1]) * temp)

def stack(sup, n, Z):
    S = np.hstack([lpa, Z]); sc = StandardScaler().fit(S[sup])
    C = 0.05 if n <= 5 else 0.1 if n <= 25 else 0.3
    return LogisticRegression(C=C, max_iter=5000).fit(sc.transform(S[sup]), ya[sup]).predict_proba(sc.transform(S))

def boost(idx, labels, n, mult=1.5, w=None):
    rounds = int(min(20 + mult * n, 400))
    bp = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=7,
              min_data_in_leaf=max(3, n // 5), feature_fraction=0.5, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=2)
    d = lgb.Dataset(Xa[idx], labels - 1, weight=w, init_score=raw_a[idx].ravel(order='F'))
    return softmax(raw_a + lgb.train(bp, d, rounds).predict(Xa, raw_score=True))

def variants(sup, n):
    Z = Xa[:, TOPK]
    pp, ps = proto(sup, Z), stack(sup, n, Z)
    pb, pb3 = boost(sup, ya[sup], n), boost(sup, ya[sup], n, mult=3.0)
    v = {'ens3': (pp + ps + pb) / 3, 'ens2': (ps + pb) / 2, 'boost': pb, 'boost_x3': pb3}
    wp = max(0.0, 1 - n / 100)                         # prototype weight fades as labels grow
    v['ensW'] = (wp * pp + ps + pb) / (wp + 2)
    v['ensW3'] = (wp * pp + ps + pb3) / (wp + 2)
    # transductive self-training: add confident unlabelled Amsterdam pixels as pseudo-labels
    base = v['ensW']; conf = base.max(1); lab = base.argmax(1) + 1
    pseudo = []
    for c in (1, 2, 3, 4):
        cand = np.where(lab == c)[0]; cand = np.setdiff1d(cand, sup)
        k = min(len(cand), max(4 * n, 100))
        pseudo.append(cand[np.argsort(conf[cand])[::-1][:k]])
    pseudo = np.concatenate(pseudo)
    idx = np.concatenate([sup, pseudo]); labs = np.concatenate([ya[sup], lab[pseudo]])
    w = np.concatenate([np.ones(len(sup)), np.full(len(pseudo), 0.3)])
    pst = boost(idx, labs, n + len(pseudo) // 8, w=w)
    v['self'] = (wp * pp + ps + pst) / (wp + 2)
    return v

res = {}
for n in (5, 25, 50, 100, 200):
    sc = {}
    for sup, te in fewshot_splits(ya, n, 5, 2):
        for k, pr in variants(sup, n).items():
            for a in (0.0, 0.5):
                p = smooth_proba(pxa, pya, pr, a, 3) if a else pr
                sc.setdefault(f'{k}|a{a}', []).append(macro_f1(ya[te], p[te].argmax(1) + 1))
    res[n] = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in sc.items()}
    print(n, sorted(res[n].items(), key=lambda kv: -kv[1][0])[:8], f'{time.time()-t0:.0f}s', flush=True)
json.dump(res, open(f'{CACHE}/exp4_{FSET}.json', 'w'), indent=1)
