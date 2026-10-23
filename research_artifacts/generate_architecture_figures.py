"""Vector schematics for the audited ACRE-Base family; no new results or inference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,FancyBboxPatch,FancyArrowPatch,Circle,Rectangle,Arc
from matplotlib.colors import to_rgb
import numpy as np,json
ROOT=Path(__file__).resolve().parent;ROOT=ROOT.parent if ROOT.name=='research_artifacts' else ROOT;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
STEM='#95A1B1'
INK='#24384B';GRAY='#68798B';BLUE='#5687AF';TEAL='#65A79D';AMBER='#D4AA61';PURPLE='#8A78A7';LINE='#D6E0E8'
def txt(ax,x,y,s,size=13,ha='left',color=INK,weight='normal',va='center'):
 return ax.text(x,y,s,fontsize=size,ha=ha,va=va,color=color,weight=weight,linespacing=1.35,zorder=10)
def arrow(ax,a,b,color=GRAY,lw=1.35,style='-|>'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=12,color=color,lw=lw,zorder=5))
def patharrow(ax,pts,color=GRAY,lw=1.1):
 for a,b in zip(pts[:-2],pts[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,zorder=4)
 arrow(ax,pts[-2],pts[-1],color,lw)
def cuboid(ax,x,y,w,h,color,depth=.11,alpha=.68):
 dx=depth;dy=depth*.6;r=np.array(to_rgb(color))
 ax.add_patch(Polygon([(x,y-h/2),(x+w,y-h/2),(x+w,y+h/2),(x,y+h/2)],closed=True,facecolor=color,edgecolor=INK,lw=.65,alpha=alpha,zorder=3))
 ax.add_patch(Polygon([(x,y+h/2),(x+dx,y+h/2+dy),(x+w+dx,y+h/2+dy),(x+w,y+h/2)],closed=True,facecolor=.68*r+.32,edgecolor=INK,lw=.65,zorder=3))
 ax.add_patch(Polygon([(x+w,y-h/2),(x+w+dx,y-h/2+dy),(x+w+dx,y+h/2+dy),(x+w,y+h/2)],closed=True,facecolor=.88*r,edgecolor=INK,lw=.65,alpha=alpha,zorder=3))
 return (x,y),(x+w+dx,y)
def rounded(ax,x,y,w,h,fill='white',edge=LINE,lw=1):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.10',fc=fill,ec=edge,lw=lw,zorder=0))
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=230,facecolor='white',bbox_inches='tight',pad_inches=.05)
 plt.close(fig)
def architecture(lang):
 id=lang=='id';fig,ax=plt.subplots(figsize=(12,6.25));fig.subplots_adjust(.01,.01,.99,.99);ax.set(xlim=(0,12),ylim=(0,6.25));ax.axis('off')
 txt(ax,.10,6.08,'(a) '+('Susunan CNN Base dan lokasi perubahan channel' if id else 'Base CNN and channel-reallocation locations'),15,weight='bold')
 y=4.55
 for j,col in enumerate(['#9CB8CD','#90B6A9','#D6B588']):cuboid(ax,.15+j*.06,y,.13,1.35,col,.10)
 txt(ax,.34,3.25,'RGB\n96×96×3',11.7,ha='center');arrow(ax,(.54,y),(.96,y))
 _,stemout=cuboid(ax,1.0,y,.18,1.35,STEM,.13)
 txt(ax,1.13,3.74,'16',12.5,ha='center');txt(ax,1.13,3.25,'Stem, s2\n48×48',11.7,ha='center')
 groups=[(1.75,1,[32,32,16],1.35,'48×48',1),(3.15,2,[32,32,24],1.12,'24×24',2),(4.55,3,[48,48,24],1.12,'24×24',1),(5.95,4,[48,48,40],.85,'12×12',2),(7.35,5,[80,80,64],.65,'6×6',2)]
 prev=stemout
 for x,b,counts,h,sp,stride in groups:
  if b>=4:rounded(ax,x-.12,3.14,1.28,2.35,'#FFF8EE' if b==4 else '#F6F2FA',AMBER if b==4 else PURPLE,.9)
  arrow(ax,prev,(x,y))
  for j,(n,c) in enumerate(zip(counts,[BLUE,TEAL,AMBER])):
   xx=x+j*.33;start,end=cuboid(ax,xx,y,.15,h,c,.10)
   txt(ax,xx+.12,3.74,str(n),12.3,ha='center')
   if j:arrow(ax,(xx-.08,y),(xx,y),lw=.85)
  end=(x+.66+.25,y)
  txt(ax,x+.45,3.25,('Blok ' if id else 'Block ')+str(b)+'\n'+sp,12.5,ha='center',weight='bold')
  txt(ax,x+.45,5.80,'s='+str(stride),12.5,ha='center',color=GRAY)
  if b in [1,3]:
   xx=x+.99;ax.add_patch(Circle((xx,y),.065,fc='white',ec=GRAY,lw=1,zorder=6));txt(ax,xx,y,'+',11,ha='center')
   patharrow(ax,[(x-.13,y),(x-.13,5.53),(xx,5.53),(xx,y+.07)],GRAY,1.1);prev=(xx+.08,y)
  else:prev=end
  if b>=4:txt(ax,x+.45,2.96,('c = 40' if b==4 else 'd = 64'),12.5,ha='center',color=AMBER if b==4 else PURPLE,weight='bold')
 arrow(ax,prev,(8.88,y));cuboid(ax,8.9,y,.17,.36,TEAL,.10)
 txt(ax,9.02,3.25,'GAP\n64',12.5,ha='center',weight='bold');arrow(ax,(9.18,y),(9.82,y))
 cuboid(ax,9.85,y,.30,.47,PURPLE,.12);txt(ax,10.02,3.25,'Linear\n64 → K',12.5,ha='center',weight='bold')
 arrow(ax,(10.3,y),(10.82,y));txt(ax,11.35,y,'K logits',13.5,ha='center',weight='bold');txt(ax,10.98,2.96,'K ∈ {5, 6, 47}',12.5,ha='center')
 txt(ax,.10,2.66,'(b) '+('Isi satu blok inverted residual (ekspansi 2×)' if id else 'Inside an inverted-residual block (expansion 2×)'),14,weight='bold')
 bx=[.6,2.43,4.34];by=1.64
 labels=[('1×1 Expand','BN + ReLU6','2Cin'),('3×3 Depthwise','BN + ReLU6','2Cin, stride s'),('1×1 Project','BN, linear','Cout')]
 for x,(a,b,c),col in zip(bx,labels,[BLUE,TEAL,AMBER]):
  rounded(ax,x,1.0,1.55,1.1,'white',col,1.3);txt(ax,x+.775,1.88,a,12,ha='center',weight='bold');txt(ax,x+.775,1.56,b,11.8,ha='center');txt(ax,x+.775,1.22,c,12.4,ha='center',color=GRAY)
 txt(ax,.12,by,'in',12.5,ha='center');arrow(ax,(.25,by),(.58,by))
 arrow(ax,(2.17,by),(2.40,by));arrow(ax,(4.0,by),(4.31,by));arrow(ax,(5.92,by),(6.26,by))
 ax.add_patch(Circle((6.40,by),.13,fc='white',ec=GRAY,lw=1));txt(ax,6.40,by,'+',14,ha='center');arrow(ax,(6.55,by),(6.9,by))
 patharrow(ax,[(.40,by),(.40,2.30),(6.40,2.30),(6.40,1.80)],GRAY)
 txt(ax,3.35,2.35,('Residual jika s=1 dan Cin=Cout' if id else 'Residual only if s=1 and Cin=Cout'),11.5,ha='center',color=GRAY,va='bottom')
 ax.plot([7.10,7.10],[.92,2.55],color=LINE,lw=1)
 txt(ax,7.38,2.37,'(c) '+('Batas lebar yang diubah' if id else 'What changes at c and d'),13.5,weight='bold')
 txt(ax,7.40,1.92,('c: keluaran Blok 4; masukan Blok 5' if id else 'c: Block 4 output; Block 5 input'),12.4)
 txt(ax,7.40,1.53,('d: keluaran Blok 5; masukan linear' if id else 'd: Block 5 output; linear-head input'),12.4)
 txt(ax,7.40,1.12,('Varian lebar: ekspansi tetap 2×' if id else 'Width variants: expansion remains 2×'),12.4,color=GRAY)
 for x,col,label in [(.2,STEM,'3×3 Stem'),(2.35,BLUE,'1×1 Expand'),(4.60,TEAL,'3×3 Depthwise'),(7.20,AMBER,'1×1 Project'),(9.55,PURPLE,'Linear head')]:
  ax.add_patch(Rectangle((x,.35),.2,.15,fc=col,ec='none'));txt(ax,x+.3,.42,label,12.5)
 save(fig,'acre_base_blocks_'+lang)
def mini_cnn(ax,x,y,scale=1):
 heights=[.58,.76,.76,.94,.65]
 for j,h in enumerate(heights):
  cuboid(ax,x+j*.28*scale,y,.09*scale,h*scale,[BLUE,TEAL,TEAL,AMBER,PURPLE][j],.06*scale)
  if j<4:arrow(ax,(x+(j*.28+.15)*scale,y),(x+(j+1)*.28*scale,y),lw=.8)
def framework(lang):
 id=lang=='id';fig,ax=plt.subplots(figsize=(12,7.4));fig.subplots_adjust(.01,.01,.99,.99);ax.set(xlim=(0,12),ylim=(0,7.4));ax.axis('off')
 txt(ax,.15,7.14,'(a) '+('Dari skor channel ke hasil model setelah retraining' if id else 'From channel assessment to retrained architecture decisions'),15,weight='bold')
 xs=[.15,3.17,6.19,9.21]
 titles=['Nilai channel pada Base','Tentukan kandidat','Retraining semua sel','Bandingkan macro-F1'] if id else ['Assess trained Base','Specify candidates','Retrain all cells','Compare macro-F1']
 for j,(x,title) in enumerate(zip(xs,titles)):
  rounded(ax,x,3.96,2.60,2.83,'white',LINE);ax.add_patch(Circle((x+.20,6.51),.13,fc=[BLUE,TEAL,AMBER,PURPLE][j],ec='none'));txt(ax,x+.20,6.51,str(j+1),11,ha='center',color='white',weight='bold');txt(ax,x+.43,6.51,title,12.3,weight='bold')
  if j<3:arrow(ax,(x+2.65,5.35),(xs[j+1]-.06,5.35),lw=1.7)
 mini_cnn(ax,.65,5.60,1.18)
 # A fixed-weight mask at a block, rather than a teacher/student icon.
 ax.add_patch(Rectangle((1.66,5.23),.25,.53,fill=False,ec=INK,lw=1,linestyle='--',zorder=7))
 txt(ax,1.45,4.82,'Masking · Taylor · BN',12.2,ha='center')
 txt(ax,1.45,4.35,('Bobot Base tetap' if id else 'Base weights held fixed'),12,ha='center',color=GRAY)
 txt(ax,4.47,5.92,'Base: (c, d)',13,ha='center',weight='bold')
 patharrow(ax,[(4.47,5.66),(4.47,5.36),(3.76,5.36),(3.76,5.06)],TEAL)
 patharrow(ax,[(4.47,5.66),(4.47,5.36),(5.18,5.36),(5.18,5.06)],TEAL)
 txt(ax,3.83,4.88,'c↑, d↓',13,ha='center',color=AMBER,weight='bold');txt(ax,5.10,4.88,'c↓, d↑',13,ha='center',color=PURPLE,weight='bold')
 txt(ax,4.47,4.35,('Base + kandidat joint\nP dan MAC dalam ±1%' if id else 'Base + joint choices\nP and MAC within ±1%'),11.9,ha='center',color=GRAY)
 mini_cnn(ax,6.95,5.85,.78);mini_cnn(ax,6.95,5.02,.78)
 txt(ax,7.49,4.37,('Inisialisasi dan latih\ndari awal' if id else 'Initialize and train\nfrom scratch'),11.8,ha='center',color=GRAY)
 rounded(ax,9.61,5.22,1.78,.72,'#F4F1F8',PURPLE,.9);txt(ax,10.50,5.58,'MAE · gain · regret',12,ha='center')
 txt(ax,10.51,4.90,('Model pilihan vs Base' if id else 'Selected model vs Base'),12.2,ha='center')
 txt(ax,10.51,4.39,('Kandidat terbaik juga dinilai' if id else 'Available best also assessed'),11.8,ha='center',color=GRAY)
 txt(ax,.15,3.57,'(b) '+('DTD prospektif: pilihan dikunci sebelum kandidat dilatih' if id else 'Prospective DTD: choices committed before candidate training'),14,weight='bold')
 yy=2.44
 steps=[('9 Base','Fit + selection'),('Diagnostik' if id else 'Diagnostics','Subset terpisah' if id else 'Separate subset'),('Kunci pilihan' if id else 'Commit choices','Hash + checkpoint'),('54 kandidat' if id else '54 candidates','6 sel per konteks' if id else '6 cells per context'),('Evaluasi' if id else 'Evaluation','Setelah 63 selesai' if id else 'After all 63 finish')]
 for j,(a,b) in enumerate(steps):
  x=.22+j*2.42;rounded(ax,x,2.0,2.01,.97,'#F5F8FA' if j!=2 else '#FFF6E8',LINE)
  txt(ax,x+1.005,2.66,a,13.1,ha='center',weight='bold');txt(ax,x+1.005,2.28,b,11.8,ha='center',color=GRAY)
  if j<4:arrow(ax,(x+2.06,yy),(x+2.35,yy))
 txt(ax,6.05,1.60,('Fit / selection / diagnostik / evaluasi memakai subset citra yang terpisah.' if id else 'Fit / selection / diagnostic / evaluation images have separate roles.'),12.5,ha='center',color=GRAY)
 ax.plot([.15,11.85],[1.25,1.25],color=LINE,lw=1)
 txt(ax,.2,.91,('279 pelatihan berbeda' if id else '279 distinct training runs'),14,weight='bold')
 labels=['36 studi awal','72 kestabilan','108 baru + 18 Base dipakai ulang','63 transfer DTD'] if id else ['36 pilot','72 stability','108 new + 18 reused Base','63 DTD transfer']
 for x,s in zip([.2,2.95,5.4,9.25],labels):txt(ax,x,.44,s,11.7,color=GRAY)
 save(fig,'acre_framework_'+lang)
for lang in ['en','id']:architecture(lang);framework(lang)
audit={'model':'ACRE-Base / AllocationNet','stem':{'kernel':3,'stride':2,'channels':16},'base_output_channels':[16,24,24,40,64],'strides':[1,2,1,2,2],'expansion_ratios':[2]*5,'expanded_channels':[32,32,48,48,80],'output_spatial_sizes':[48,24,24,12,6],'residual_blocks':[1,3],'head':'global average pooling followed by a biased linear layer','classes':[5,6,47],'no_msf_branch':True,'meaning':'Feature blocks are schematic; dimensions and channel labels are explicit. No semantic-stage or hardware-efficiency claim.'}
(ROOT/'FIGURE_ARCHITECTURE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Generated bilingual vector framework and CNN-block figures.')
