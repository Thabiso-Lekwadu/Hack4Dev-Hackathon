"""Experiment 1 — which feature groups help Madrid CV, zero-shot and 25-shot transfer?"""
import numpy as np, lightgbm as lgb, time, json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from common import *

mad, ams, G = load()
ym, ya = mad.age_class.values, ams.age_class.values
P = dict(n_estimators=400, learning_rate=0.08, num_leaves=63, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
         class_weight='balanced', random_state=SEED, n_jobs=2, verbose=-1)
sets = {
 'baseline':            (G['baseline'], False),
 'baseline+citynorm':   (G['baseline'], True),
 'new(no spatial)+cn':  (G['anomaly'] + G['change'] + G['meta'], True),
 'all-no-spatial+cn':   (G['baseline'] + G['anomaly'] + G['change'] + G['meta'], True),
 'all+cn':              (G['baseline'] + G['anomaly'] + G['change'] + G['meta'] + G['spatial'], True),
}
res = {}
for name, (cols, cn) in sets.items():
    t0 = time.time()
    Xm = mad[cols].values; Xa = ams[cols].values
    if cn: Xm, Xa = city_norm(Xm), city_norm(Xa)
    f1s = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xm, ym):
        m = lgb.LGBMClassifier(**P).fit(Xm[tr], ym[tr]); f1s.append(macro_f1(ym[te], m.predict(Xm[te])))
    m = lgb.LGBMClassifier(**P).fit(Xm, ym)
    pa = m.predict_proba(Xa); zs = macro_f1(ya, m.classes_[pa.argmax(1)])
    lp = np.log(pa + 1e-6)
    shots = {}
    for n in (5, 25, 100):
        s = []
        for sup, te in fewshot_splits(ya, n, 5, 1):
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(lp[sup], ya[sup])
            s.append(macro_f1(ya[te], lr.predict(lp[te])))
        shots[n] = round(float(np.mean(s)), 4)
    res[name] = dict(n_feat=len(cols), madrid_cv=round(float(np.mean(f1s)), 4), zero_shot=round(zs, 4), recal=shots, sec=int(time.time() - t0))
    print(name, res[name], flush=True)
json.dump(res, open(f'{CACHE}/exp1.json', 'w'), indent=1)
