import sys,time,json,numpy as np,pandas as pd
from lib import *
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
names=[d[0] for d in Descriptors._descList]+[f'morgan_count_{i}' for i in range(512)]
def load(ds):
    if ds=='tox21':
        df=pd.read_csv('tox21.csv'); X,keep,sc=featurize(df.smiles.values)
        d=df.iloc[keep].reset_index(drop=True); tasks=[c for c in df.columns if c.startswith(('NR-','SR-'))]
        return X,sc,{t:d[t].values for t in tasks}
    df=pd.read_csv(ds+'.csv'); X,keep,sc=featurize(df.smiles.values); d=df.iloc[keep].reset_index(drop=True)
    return X,sc,({'CT_TOX':d['CT_TOX'].values} if ds=='clintox' else {'Hepatobiliary':d['Hepatobiliary disorders'].values})
import os; os.makedirs('results',exist_ok=True)
out=open('results/results_rerun.csv','a'); feats=open('results/features_rerun.jsonl','a')
for ds in sys.argv[1:]:
    X,sc,tasks=load(ds)
    for task,y in tasks.items():
        ok=~np.isnan(y); Xo=X[ok]; yo=y[ok].astype(int); so=sc[ok]
        for seed in range(3):
            tr,te=scaffold_split(so,seed)
            if yo[te].min()==yo[te].max() or yo[tr].min()==yo[tr].max(): continue
            Xtr,Xte=prep(Xo[tr].copy(),Xo[te].copy())
            kept=np.where(np.nanstd(Xo[tr],0)>1e-8)[0]  # approx; same mask as prep (finite-safe)
            for noise in (0.0,0.1,0.2):
                rng=np.random.RandomState(1000+seed); ytr=yo[tr].copy()
                flip=rng.rand(len(ytr))<noise; ytr[flip]=1-ytr[flip]
                models={'LASSO':lambda:lasso_logit(Xtr,ytr,seed),
                        'SLTS+LASSO (h=0.9)':lambda:slts_lasso(Xtr,ytr,0.9,seed),
                        'SLTS+LASSO (h=0.75)':lambda:slts_lasso(Xtr,ytr,0.75,seed),
                        'Random Forest':lambda:RandomForestClassifier(200,n_jobs=2,random_state=seed,class_weight='balanced_subsample',min_samples_leaf=2).fit(Xtr,ytr),
                        'Gradient boosting':lambda:HistGradientBoostingClassifier(max_iter=150,random_state=seed).fit(Xtr,ytr)}
                for mn,f in models.items():
                    m=f(); p=m.decision_function(Xte) if hasattr(m,'decision_function') else m.predict_proba(Xte)[:,1]
                    nz=int(np.sum(m.coef_!=0)) if hasattr(m,'coef_') else -1
                    out.write(f"{ds},{task},{seed},{noise},{mn},{roc_auc_score(yo[te],p):.4f},{average_precision_score(yo[te],p):.4f},{nz},{len(tr)},{len(te)},{yo[te].mean():.4f}\n"); out.flush()
                    if noise==0 and seed==0 and mn in('LASSO','SLTS+LASSO (h=0.9)') and nz>0:
                        c=m.coef_[0]; idx=np.argsort(-np.abs(c))[:8]; ke=np.where(np.std(prep(Xo[tr].copy(),Xo[te].copy())[0],0)>=0)[0]
                        feats.write(json.dumps({'ds':ds,'task':task,'model':mn,'n_nonzero':nz,'top':[[int(i),float(c[i])] for i in idx]})+"\n"); feats.flush()
print('DONE')
