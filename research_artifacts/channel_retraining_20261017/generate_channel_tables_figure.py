from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
A=Path(__file__).resolve().parent
S=A.parent.parent
s=json.loads((A/'analysis_summary.json').read_text());dslabels={'warpc':'WaRP-C','trashnet':'TrashNet','dtd':'DTD'}
rows=[]
for z in s['local_taylor']:
 rows.append(dslabels[z['dataset']]+f" & {z['taylor_mae']:.3f} / {z['zero_mae']:.3f} & {z['retrained_ce_mae']:.3f} / {z['zero_retrained_ce_mae']:.3f} "+r'\\')
(S/'tables/channel_local_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for ds in dslabels:
 p=next(z for z in s['projection'] if z['dataset']==ds and z['target']=='f1' and z['method']=='mask_f1')
 task=next(z for z in s['averaged_response_projection'] if z['dataset']==ds and z['target']=='f1' and z['level']=='task')
 c,d=[next(z for z in s['bidirectional'] if z['dataset']==ds and z['target']=='f1' and z['axis']==axis) for axis in ['c','d']]
 rows.append(dslabels[ds]+f" & {p['mean_minimum_rmse']:.3f} & {task['minimum_rmse']:.3f} & {c['same_sign']}/9 & {d['same_sign']}/9 "+r'\\')
(S/'tables/channel_projection_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for z in s['projection']:
 lab={'mask_f1':'F1 mask','architecture_ridge':'Ridge','mask_macro_ce':'Loss mask','taylor_signed':'Signed Taylor'}[z['method']]
 rows.append(dslabels[z['dataset']]+f" & {z['target'].replace('_',' ')} & {lab} & {z['mean_actual_rmse']:.5f} & {z['mean_minimum_rmse']:.5f} & {z['mean_zero_rmse']:.5f} & {100*z['pooled_minimum_fraction_of_sse']:.2f} "+r'\\')
(S/'tables/channel_projection_full_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for z in s['intervention']:
 lab={'c_grow':r'$c\uparrow$','d_shrink':r'$d\downarrow$','joint_c_grow_d_shrink':r'$c\uparrow,d\downarrow$','c_shrink':r'$c\downarrow$','d_grow':r'$d\uparrow$','joint_c_shrink_d_grow':r'$c\downarrow,d\uparrow$'}[z['candidate']]
 rows.append(dslabels[z['dataset']]+f" & {lab} & {z['mean_gain_pp']:+.3f} & {z['proxy_mae']:.3f} & {z['zero_mae']:.3f} & {z['sign_agreement']}/9 "+r'\\')
(S/'tables/channel_intervention_rows.tex').write_text('\n'.join(rows)+'\n')
# New figure: local score fidelity and bidirectional retraining response.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
fig,axs=plt.subplots(1,3,figsize=(12.4,3.7));blue='#326B91';orange='#BA7632';teal='#438D8C';gray='#6C7480'
x=np.arange(3);local=s['local_taylor'];ax=axs[0]
ax.bar(x-.17,[z['taylor_mae']/z['zero_mae'] for z in local],.33,color=blue,label='Frozen mask response')
ax.bar(x+.17,[z['retrained_ce_mae']/z['zero_retrained_ce_mae'] for z in local],.33,color=orange,label='Retrained width gain')
ax.axhline(1,color=gray,ls='--',lw=1);ax.set_xticks(x,[dslabels[z['dataset']] for z in local],fontsize=10);ax.set_ylabel('Taylor MAE / zero-prediction MAE');ax.set_title('(a) Local relevance and retraining',fontsize=11);ax.legend(fontsize=8,frameon=False,loc='lower left');ax.set_ylim(0,1.42)
bidir=list(csv.DictReader((A/'bidirectional_contexts.csv').open()))
for ax,axis,title in zip(axs[1:],['c','d'],['(b) Width c: both retrained directions','(c) Width d: both retrained directions']):
 for ds,col in [('warpc',blue),('trashnet',orange),('dtd',teal)]:
  a=[z for z in bidir if z['dataset']==ds and z['target']=='f1' and z['axis']==axis]
  ax.scatter([float(z['growth_gain']) for z in a],[float(z['shrink_gain']) for z in a],s=27,color=col,label=dslabels[ds],alpha=.8)
 ax.axhline(0,color=gray,lw=1);ax.axvline(0,color=gray,lw=1);ax.set_xlabel('Growth gain (pp)');ax.set_ylabel('Shrinkage gain (pp)');ax.set_title(title,fontsize=11);ax.grid(alpha=.12);ax.legend(fontsize=8,frameon=False,loc='best')
fig.tight_layout();fig.savefig(S/'figures/channel_retraining_boundary.pdf');fig.savefig(S/'figures/channel_retraining_boundary.png',dpi=160);plt.close(fig)
