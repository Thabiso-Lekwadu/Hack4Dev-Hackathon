"""Baseline (Notebook 3/4: 60 features, Madrid-fit StandardScaler, RandomForest 500, prototypes)
under the SAME random and blocked splits as leakage_audit.py, for a like-for-like comparison."""
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])  # repository root
import json, time, numpy as np, pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score
SEED, BLOCK, BUFFER = 42, 67, 12
FEAT = ROOT + '/data/features'
g = json.load(open(f'{FEAT}/feature_groups.json')); B = g['baseline']
mad = pd.read_parquet(f'{FEAT}/features_madrid.parquet'); ams = pd.read_parquet(f'{FEAT}/features_amsterdam.parquet')
ym, ya = mad.age_class.values.astype(int), ams.age_class.values.astype(int)
sc = StandardScaler().fit(mad[B].values)                       # Notebook 3: fit on Madrid, apply to both
Xm, Xa = sc.transform(mad[B].values), sc.transform(ams[B].values)
f1 = lambda y, p: f1_score(y, p, average='macro')
def buf(px, py, te):
    x, y = px - px.min(), py - py.min(); gr = np.zeros((y.max() + 1, x.max() + 1), bool); gr[y[te], x[te]] = True
    return maximum_filter(gr, size=2 * BUFFER + 1)[y, x]
R = {}; t0 = time.time()
px, py = mad.px_key.values, mad.py_key.values; tile = (px // BLOCK) * 100000 + (py // BLOCK)
splits = {'random': list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xm, ym)), 'blocked': []}
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xm, ym, tile):
    splits['blocked'].append((tr[~buf(px, py, te)[tr]], te))
for mode, sp in splits.items():
    s = [f1(ym[te], RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=SEED, n_jobs=-1).fit(Xm[tr], ym[tr]).predict(Xm[te])) for tr, te in sp]
    R[f'baseline_madrid_{mode}'] = (round(float(np.mean(s)), 4), round(float(np.std(s)), 4))
    print('baseline madrid', mode, R[f'baseline_madrid_{mode}'], f'[{time.time()-t0:.0f}s]', flush=True)
apx, apy = ams.px_key.values, ams.py_key.values; atile = (apx // BLOCK) * 100000 + (apy // BLOCK)
def proto(sup, q):
    P = np.stack([Xa[sup][ya[sup] == c].mean(0) for c in (1, 2, 3, 4)])
    return np.linalg.norm(Xa[q][:, None] - P[None], axis=2).argmin(1) + 1
for mode in ('random', 'blocked'):
    res = {}
    for n in (5, 25, 50, 100, 200):
        s = []
        for rep in range(2):
            rng = np.random.default_rng(SEED + rep)
            spl = (StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(Xa, ya) if mode == 'random'
                   else StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep).split(Xa, ya, atile))
            for tr, te in spl:
                if mode == 'blocked': tr = tr[~buf(apx, apy, te)[tr]]
                sup = np.concatenate([rng.choice(tr[ya[tr] == c], n, replace=False) for c in (1, 2, 3, 4)])
                s.append(f1(ya[te], proto(sup, te)))
        res[n] = (round(float(np.mean(s)), 4), round(float(np.std(s)), 4))
    R[f'baseline_amsterdam_{mode}'] = res; print('baseline amsterdam', mode, res, flush=True)
L = json.load(open(ROOT + '/data/cache/leakage_report.json')); L.update(R)
json.dump(L, open(ROOT + '/data/cache/leakage_report.json', 'w'), indent=1); print('DONE', flush=True)
