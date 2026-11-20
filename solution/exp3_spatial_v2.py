"""Experiment 3 — does richer spatial context (v2) beat v1? Same protocol as exp1 + the exp2 ensemble at 25 shots."""
import numpy as np, lightgbm as lgb, json, pickle
from sklearn.model_selection import StratifiedKFold
from common import *
from spatial_v2 import add_spatial_v2

mad, ams, G = load()
for df in (mad, ams):
    pass
sm, sa = add_spatial_v2(mad), add_spatial_v2(ams)
mad = pd.concat([mad, sm], axis=1); ams = pd.concat([ams, sa], axis=1)
G['spatial_v2'] = list(sm.columns)
mad.to_parquet(f'{CACHE}/features_madrid_v2.parquet'); ams.to_parquet(f'{CACHE}/features_amsterdam_v2.parquet')
pickle.dump(G, open(f'{CACHE}/feature_groups_v2.pkl', 'wb'))
print('v2 spatial features:', len(sm.columns), flush=True)

ym, ya = mad.age_class.values, ams.age_class.values
P = dict(n_estimators=400, learning_rate=0.08, num_leaves=63, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
         class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1)
core = G['baseline'] + G['anomaly'] + G['change'] + G['meta']
sets = {'core+v2': core + G['spatial_v2'], 'core+v1+v2': core + G['spatial'] + G['spatial_v2']}
res = {}
for name, cols in sets.items():
    Xm, Xa = city_norm(mad[cols].values), city_norm(ams[cols].values)
    f1s = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xm, ym):
        m = lgb.LGBMClassifier(**P).fit(Xm[tr], ym[tr]); f1s.append(macro_f1(ym[te], m.predict(Xm[te])))
    m = lgb.LGBMClassifier(**P).fit(Xm, ym)
    zs = macro_f1(ya, m.predict(Xa))
    res[name] = dict(n_feat=len(cols), madrid_cv=round(float(np.mean(f1s)), 4), zero_shot=round(zs, 4))
    print(name, res[name], flush=True)
json.dump(res, open(f'{CACHE}/exp3.json', 'w'), indent=1)
