import pandas as pd,numpy as np
import os; d=pd.read_csv('results/results.csv' if os.path.exists('results/results.csv') else 'results/results_rerun.csv')
d['grp']=d.dataset+'/'+d.task
print(len(d),'rows;',d.grp.nunique(),'tasks; seeds',sorted(d.seed.unique()))
# complete cells only
cnt=d.groupby(['grp','seed','noise']).model.count(); full=cnt[cnt==5].reset_index()[['grp','seed','noise']]
d=d.merge(full,on=['grp','seed','noise'])
def tab(df,name):
    t=df.groupby(['model','noise']).roc_auc.mean().unstack().round(3); 
    t['drop_0to20']=(t[0.0]-t[0.2]).round(3); t['drop_0to10']=(t[0.0]-t[0.1]).round(3)
    print('\n==',name,'mean ROC-AUC (n=%d task-seed cells)'%df.groupby(['grp','seed']).ngroups); print(t)
tab(d,'ALL')
tab(d[d.dataset=='tox21'],'Tox21 (12 tasks)')
for ds in ['clintox','sider']:
    if (d.dataset==ds).any(): tab(d[d.dataset==ds],ds)
# paired: slts vs lasso per cell
p=d.pivot_table(index=['grp','seed','noise'],columns='model',values='roc_auc')
for noise in (0.0,0.1,0.2):
    q=p.xs(noise,level='noise')
    for m in ['SLTS+LASSO (h=0.9)','SLTS+LASSO (h=0.75)']:
        diff=q[m]-q['LASSO']; print(f'noise {noise}: {m} vs LASSO: mean diff {diff.mean():+.3f}, wins {(diff>0).sum()}/{len(diff)}')
    diff=q['SLTS+LASSO (h=0.9)']-q['Random Forest']; print(f'noise {noise}: SLTS0.9 vs RF mean diff {diff.mean():+.3f}, wins {(diff>0).sum()}/{len(diff)}')
nz=d[d.noise==0].groupby('model').n_nonzero.mean().round(1); print('\nmean nonzero coefs (clean):\n',nz)
