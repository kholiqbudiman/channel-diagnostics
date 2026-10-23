from pathlib import Path
import json,csv,statistics,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
S=Path(__file__).resolve().parent.parent;T=S/'research_artifacts/proxy_transfer_20261017';C=T/'completed_20261017';O=T/'old_comparison_complete/old_comparison'
load=lambda p:json.loads(p.read_text())
rows=lambda p:list(csv.DictReader(p.open()))
new=load(C/'verified_results_snapshot.json');old=load(S/'research_artifacts/directed_20261017/review/results_snapshot.json')
summary=load(C/'dtd_comparison/summary.json');osummary=load(O/'summary.json')
methods=['mask_f1','mask_macro_ce','taylor_signed','taylor_squared','bn_gamma','architecture_ridge','choose_base']
labels={'mask_f1':'F1 mask','mask_macro_ce':'Loss mask','taylor_signed':'Signed Taylor','taylor_squared':'Squared Taylor','bn_gamma':'BN scale','architecture_ridge':'Architecture ridge','choose_base':'Retain Base'}
dslabels={'warpc':'WaRP-C','trashnet':'TrashNet','dtd':'DTD'}
order=['base','c_grow','d_shrink','joint_c_grow_d_shrink','c_shrink','d_grow','joint_c_shrink_d_grow']
short=['Base',r'$c\uparrow$',r'$d\downarrow$',r'$c\uparrow,d\downarrow$',r'$c\downarrow$',r'$d\uparrow$',r'$c\downarrow,d\uparrow$']
lookup={(x['dataset'],x['method']):x for x in osummary['methods']+summary['methods']}
put=lambda name,a:(S/'tables'/name).write_text('\n'.join(a)+'\n')
rr=[]
for m in methods:
 v=[lookup[d,m]['mean_selected_gain_pp'] for d in dslabels]
 z=lookup['dtd',m];rho='---' if z['mean_spearman_f1'] is None else f"{z['mean_spearman_f1']:+.3f}"
 rr.append(labels[m]+' & '+' & '.join(f'${x:+.3f}$' for x in v)+f" & {z['mean_regret_pp']:.3f} & {rho} "+r'\\')
put('empirical_policy_rows.tex',rr)
rr=[];costall=[]
for ds in dslabels:
 a=new if ds=='dtd' else old
 configs={r['job']['model']['name']:r for r in a if r['job']['dataset']==ds}
 for n,sl in zip(order,short):
  x=configs[n];c=x['cost'];ch=x['job']['model']['channels'];row=dslabels[ds]+f' & ({ch[2]},{ch[3]}) & '+f"{c['params']:,} & {c['macs']/1e6:.6f} "+r'\\'
  if n in ['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow']:rr.append(row)
  if ds=='dtd':costall.append(sl+f' & ({ch[2]},{ch[3]}) & '+f"{c['params']:,} & {c['macs']/1e6:.6f} "+r'\\')
put('empirical_joint_costs.tex',rr);put('dtd_cost_rows.tex',costall)
oldinter=rows(O/'all_interactions.csv');newinter=rows(C/'dtd_comparison/all_interactions.csv');inter=oldinter+newinter
rr=[]
for ds in dslabels:
 a=new if ds=='dtd' else old
 idx={(r['review_split'],r['job']['seed'],r['job']['model']['name']):r for r in a if r['job']['dataset']==ds}
 for direction,model,sl in [('c_up_d_down','joint_c_grow_d_shrink',r'$c\uparrow,d\downarrow$'),('c_down_d_up','joint_c_shrink_d_grow',r'$c\downarrow,d\uparrow$')]:
  ints=[z for z in inter if z['dataset']==ds and z['direction']==direction]
  vals=[float(z['interaction_f1_pp']) for z in ints];acc=[float(z['interaction_accuracy_pp']) for z in ints]
  gains=[r['probe_metrics']['macro_f1_pct']-idx[sp,seed,'base']['probe_metrics']['macro_f1_pct'] for (sp,seed,n),r in idx.items() if n==model]
  rr.append(dslabels[ds]+' & '+sl+' & '+' & '.join(f'${v:+.2f}$' for v in [np.mean(gains),np.mean(vals),np.mean(acc)])+f' & {sum(v<0 for v in vals)}/9 '+r'\\')
put('empirical_interaction_rows.tex',rr)
# Context-specific DTD results and intervals.
idx={(r['review_split'],r['job']['seed'],r['job']['model']['name']):r for r in new}
boot={(r['split'],r['seed']):r for r in load(C/'conditional_bootstrap.json')['rows']}
dec=rows(C/'dtd_comparison/all_decisions.csv');di={(int(r['split']),int(r['seed']),r['method']):r for r in dec}
rr=[];hr=[];ir=[];allr=[]
for sp in [6301,6302,6303]:
 for seed in [3901,3902,3903]:
  b=idx[sp,seed,'base']['probe_metrics']['macro_f1_pct'];f=di[sp,seed,'mask_f1'];ce=di[sp,seed,'mask_macro_ce'];z=boot[sp,seed]
  code=lambda n: 'B' if n=='base' else ('U' if n=='joint_c_grow_d_shrink' else 'R')
  delta=float(ce['gain_pp'])-float(f['gain_pp']);ci=z['ce_minus_f1_policy_interval_pp']
  rr.append(f"{sp} & {seed} & {code(f['selected'])} & {code(ce['selected'])} & {delta:+.3f} & [{ci[0]:+.3f},{ci[1]:+.3f}] & {float(ce['gain_pp']):+.3f} "+r'\\')
  g=[idx[sp,seed,n]['probe_metrics']['macro_f1_pct']-b for n in ['joint_c_grow_d_shrink','joint_c_shrink_d_grow']]
  hr.append(f'{sp} & {seed} & {b:.3f} & {g[0]:+.3f} & {g[1]:+.3f} & {max(0,*g):.3f} '+r'\\')
  for direction,sl in [('c_up_d_down',r'$c\uparrow,d\downarrow$'),('c_down_d_up',r'$c\downarrow,d\uparrow$')]:
   t=next(r for r in newinter if int(r['split'])==sp and int(r['seed'])==seed and r['direction']==direction)
   ci=z['interaction_intervals'][direction]
   ir.append(f'{sp} & {seed} & '+sl+f" & {float(t['interaction_f1_pp']):+.3f} & [{ci[0]:+.3f},{ci[1]:+.3f}] "+r'\\')
  for n,sl in zip(order,short):
   x=idx[sp,seed,n];met=x['probe_metrics'];allr.append(f"{sp} & {seed} & {sl} & {met['macro_f1_pct']:.3f} & {met['accuracy_pct']:.3f} & {met['macro_f1_pct']-b:+.3f} & {x['best_epoch']} "+r'\\')
put('dtd_primary_rows.tex',rr);put('dtd_headroom_rows.tex',hr);put('dtd_interaction_rows.tex',ir);put('dtd_run_rows.tex',allr)
# Vector figures, exact observations and descriptive means; no synthetic errors.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
blue='#326B91';orange='#BA7632';teal='#438D8C';gray='#6C7480';purple='#826DA1';light='#EDF3F7'
fig=plt.figure(figsize=(12,6.0));ax=fig.add_axes([.02,.02,.96,.96]);ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off')
def box(x,y,w,h,title,body,fc=light):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.09',facecolor=fc,edgecolor=blue,lw=1))
 ax.text(x+.12,y+h-.19,title,weight='bold',va='top',fontsize=12,color=blue)
 ax.text(x+.12,y+h-.58,body,va='top',fontsize=11,linespacing=1.6)
def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=13,color=gray,lw=1.3))
box(.15,4.52,2.45,1.15,'1  Pilot','36 new runs\nCandidate development')
box(3.15,4.52,2.45,1.15,'2  Stability','72 new runs\nRankings, splits, classes')
box(6.15,4.52,2.45,1.15,'3  Directed responses','108 new + 18 reused Base\nSeparate + joint changes')
box(9.15,4.52,2.6,1.15,'4  DTD transfer','63 new runs\nFixed diagnostics and choices')
for x in [2.62,5.62,8.62]:arrow(x,5.08,x+.5,5.08)
ax.text(.12,4.05,'Prospective DTD: four separate data roles',fontsize=11,weight='bold',color=blue)
box(.15,2.5,2.45,1.18,'Fit + selection','Train nine Base models\nSelect checkpoints',fc='#F0F2F5')
box(3.15,2.5,2.45,1.18,'Diagnostic subset','Five diagnostic mappings\nScore matched candidates',fc='#EBF4F2')
box(6.15,2.5,2.45,1.18,'Commit decisions','Hash predictions + choices\nBefore 54 new runs',fc='#FFF3E6')
box(9.15,2.5,2.6,1.18,'Final evaluation','Open after all 63 complete\nCompare every policy',fc='#F1EDF7')
for x in [2.62,5.62,8.62]:arrow(x,3.04,x+.5,3.04)
ax.text(6.22,2.08,'Retrain all six nonreference cells per context',color=gray,fontsize=11)
for x,t,bod in [(.15,'Prediction fidelity','MAE in stated units; candidate ranks'),(4.15,'Decision utility','Selected gain; regret versus available best'),(8.15,'Candidate headroom','Does either joint candidate improve Base?')]:
 box(x,.35,3.55,1.07,t,bod,fc='white')
ax.text(.15,1.73,'Three complementary assessments',fontsize=11,weight='bold',color=blue)
fig.savefig(S/'figures/empirical_overview.pdf');fig.savefig(S/'figures/empirical_overview.png',dpi=160);plt.close(fig)
# Paired policy gains.
all_dec=rows(O/'all_decisions.csv')+dec
fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,ds in zip(axs,dslabels):
 ax.axvline(0,color=gray,lw=1)
 for i,m in enumerate(methods):
  vals=np.array([float(r['gain_pp']) for r in all_dec if r['dataset']==ds and r['method']==m]);assert len(vals)==9
  ax.scatter(vals,i+np.linspace(-.12,.12,len(vals)),s=16,color=blue if m!='choose_base' else gray,alpha=.5)
  ax.scatter(np.mean(vals),i,marker='D',s=38,color=orange,zorder=4)
 ax.set_title(dslabels[ds]+(' (prospective)' if ds=='dtd' else ' (development)'),fontsize=12)
 ax.set_yticks(range(7),[labels[m] for m in methods]);ax.invert_yaxis();ax.set_xlabel('Selected gain over Base (pp)');ax.grid(axis='x',alpha=.13)
fig.tight_layout();fig.savefig(S/'figures/empirical_policy_gains.pdf');plt.close(fig)
# DTD headroom top candidates, bottom mean policy gains.
fig,axs=plt.subplots(2,1,figsize=(11,5.1),gridspec_kw={'height_ratios':[1.5,1]})
head=sorted(load(C/'candidate_headroom_review.json'),key=lambda x:(x['split'],x['seed']))
x=np.arange(9);ax=axs[0];ax.axhline(0,color=gray,lw=1)
for j,col,label in [(0,blue,r'$c\uparrow,d\downarrow$'),(1,teal,r'$c\downarrow,d\uparrow$')]:
 ax.scatter(x+(-.07 if j==0 else .07),[h['joint_gains_pp'][j] for h in head],color=col,s=38,label=label)
ax.plot(x,[h['oracle_gain_pp'] for h in head],color=orange,marker='D',markersize=4,lw=1,label='Hindsight best, including Base')
ax.set_xticks(x,[f"{h['split']}\n{h['seed']}" for h in head]);ax.set_ylabel('Candidate gain (pp)');ax.legend(loc='lower left',ncol=3,frameon=False,fontsize=10)
ax.set_title('(a) Only two of eighteen joint effects improve on Base',loc='left',fontsize=12)
ax=axs[1];v=[lookup['dtd',m]['mean_selected_gain_pp'] for m in methods];ax.bar(range(7),v,color=[blue]*6+[gray],width=.65)
ax.axhline(0,color=gray,lw=1);ax.axhline(np.mean([h['oracle_gain_pp'] for h in head]),color=orange,lw=1.3,ls='--',label='Mean oracle headroom: +0.612 pp')
ax.set_xticks(range(7),[labels[m] for m in methods],fontsize=10);ax.set_ylabel('Mean gain (pp)');ax.set_title('(b) All diagnostic choices have negative mean gain',loc='left',fontsize=12);ax.legend(loc='lower left',frameon=False,fontsize=10)
fig.tight_layout();fig.savefig(S/'figures/empirical_headroom.pdf');plt.close(fig)
# All individual and split means for interactions.
fig,axs=plt.subplots(1,3,figsize=(11.6,3.5),sharey=True)
colors=[blue,orange,teal]
for ax,ds in zip(axs,dslabels):
 ax.axhline(0,color=gray,lw=1);spvals=sorted({int(z['split']) for z in inter if z['dataset']==ds})
 for j,direction in enumerate(['c_up_d_down','c_down_d_up']):
  vv=[]
  for k,sp in enumerate(spvals):
   vals=[float(z['interaction_f1_pp']) for z in inter if z['dataset']==ds and z['direction']==direction and int(z['split'])==sp];vv+=vals
   xx=j+(k-1)*.19;ax.scatter(xx+np.array([-.035,0,.035]),vals,color=colors[k],s=22,alpha=.7);ax.plot([xx-.065,xx+.065],[np.mean(vals)]*2,color=colors[k],lw=2)
  ax.scatter(j,np.mean(vv),marker='D',color='black',s=34,zorder=5)
 ax.set_xticks([0,1],[r'$c\uparrow,d\downarrow$',r'$c\downarrow,d\uparrow$']);ax.set_title(dslabels[ds]);ax.set_xlim(-.45,1.45);ax.grid(axis='y',alpha=.13)
axs[0].set_ylabel('Finite macro-F1 interaction (pp)')
fig.tight_layout();fig.savefig(S/'figures/empirical_interactions.pdf');plt.close(fig)

print('Regenerated empirical tables and four vector figures from preserved records.')
