"""Post-completion descriptive audit; never trains or changes the locked pilot."""
from pathlib import Path
from collections import defaultdict
import json, csv, hashlib, statistics, os
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
snapshot=json.loads((HERE/'verified_pilot_snapshot.json').read_text())
previous=json.loads((HERE/'complete_pilot_analysis.json').read_text())
assert snapshot['audit']['sources']==previous['sources']
rows=snapshot['results']; summary=[]; perclass=[]; audit={'completed':len(rows),'unit_tests_passed':10,'source_hashes_match_previous_analysis':True,'initial_weights_base_vs_no_residual_equal_pairs':4,'split_checks':{}}
for ds in ['warpc','trashnet']:
 manifest=json.loads((ROOT/f'data_{ds}.json').read_text())
 groups=defaultdict(set)
 for split,indices in manifest['split'].items():
  for i in indices: groups[manifest['image_sha256'][manifest['samples'][i][0]]].add(split)
 assert not any(len(v)>1 for v in groups.values())
 audit['split_checks'][ds]={'cross_split_identical_image_hash_groups':0,'scope':'Exact bytes only; not a near-duplicate or source-object grouping audit.'}
 subset=[r for r in rows if r['job']['dataset']==ds]
 base={r['job']['seed']:r for r in subset if r['job']['model']['name']=='base'}
 for name in sorted({r['job']['model']['name'] for r in subset}):
  runs=sorted([r for r in subset if r['job']['model']['name']==name],key=lambda r:r['job']['seed'])
  assert len(runs)==2
  vals=[r['probe_metrics']['macro_f1_pct'] for r in runs]
  deltas=[v-base[r['job']['seed']]['probe_metrics']['macro_f1_pct'] for v,r in zip(vals,runs)]
  cost=runs[0]['cost']; bc=base[1901]['cost']
  summary.append(dict(dataset=ds,model=name,n=2,params=cost['params'],macs=cost['macs'],parameter_change_pct=100*(cost['params']/bc['params']-1),mac_change_pct=100*(cost['macs']/bc['macs']-1),probe_f1_mean=statistics.mean(vals),probe_f1_sd=statistics.stdev(vals),gain_mean_pp=statistics.mean(deltas),gain_seed1901_pp=deltas[0],gain_seed1902_pp=deltas[1],selection_f1_mean=statistics.mean(r['selection_macro_f1_pct'] for r in runs)))
  for r in runs:
   best=max(r['history'],key=lambda e:e['selection_macro_f1_pct'])
   assert best['epoch']==r['best_epoch'] and best['selection_macro_f1_pct']==r['selection_macro_f1_pct']
   cm=np.array(r['probe_metrics']['confusion_matrix']);support=cm.sum(1)
   f1=200*np.diag(cm)/(support+cm.sum(0))
   for i,cl in enumerate(r['classes']):perclass.append(dict(dataset=ds,model=name,seed=r['job']['seed'],class_name=cl,support=int(support[i]),f1_pct=float(f1[i]),recall_pct=r['probe_metrics']['class_recall_pct'][i]))
validation=[]
for mode in ['architecture_only','frequency_only','class_agnostic','class_pair']:
 for axis in ['allocation','dataset']:
  folds=[v for v in previous['surrogate_validation'] if v['mode']==mode and v['heldout_axis']==axis]
  predictions=[p for f in folds for p in f['predictions']]
  choices=[s for f in folds for s in f['selection']]
  validation.append(dict(mode=mode,holdout=axis,folds=len(folds),predictions=len(predictions),mae_pp=statistics.mean(abs(p['predicted_macro_f1_pct']-p['observed_macro_f1_pct']) for p in predictions),mean_selected_gain_pp=statistics.mean(s['gain_vs_manual_base_pp'] for s in choices) if choices else None,mean_oracle_regret_pp=statistics.mean(s['development_oracle_regret_pp'] for s in choices) if choices else None))
for name,data in [('architecture_summary.csv',summary),('classwise_probe.csv',perclass),('surrogate_summary.csv',validation)]:
 with (HERE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
audit['checkpoint_selection_history_pass']=36
(HERE/'review_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
lines=['PEMERIKSAAN ACRE — PILOT LENGKAP 16 OKTOBER 2026',
'','KESIMPULAN',
'Pilot selesai 36/36 tanpa kegagalan. Audit artefak lolos. Hasilnya memberi',
'dasar untuk memilih eksperimen berikutnya, tetapi belum menunjukkan bahwa',
'prototipe pemilih alokasi berbasis pasangan kelas lebih baik daripada kontrol.',
'Jangan menaikkan klaim novelty atau menghidupkan kembali klaim minoritas legacy.',
'','RUANG LINGKUP',
'9 konfigurasi x 2 seeds (1901/1902) x 2 dataset; satu split stratified 4301.',
'Checkpoint dipilih dengan selection macro-F1. Angka di bawah adalah PROBE',
'development dari checkpoint tersebut. Probe sudah terekspos untuk pengembangan',
'metode dan bukan final test independen. Tidak ada training baru dalam pemeriksaan ini.',
'','AUDIT YANG DILAKUKAN',
'- Audit CPU load_complete dijalankan ulang pada seluruh cohort di server.',
'- Hash input terkunci, best.pt, initial_state.pt, dan split sesuai.',
'- Urutan label probe sesuai manifest; logits -> confusion -> macro-F1 cocok.',
'- Seluruh 36 checkpoint terpilih cocok dengan maksimum pertama history selection.',
'- Hash 36 result/prediction/checkpoint cocok dengan analisis otomatis sebelumnya.',
'- 10 pemeriksaan unit analisis PASS. Dua sempat terhalang direktori sementara',
'  pada sandbox read-only, kemudian seluruhnya lulus setelah izin eksekusi.',
'- Bobot awal Base vs no-residual identik: 4/4 pasangan, 98 tensor per pasangan.',
'- Tidak ada hash citra identik yang melintasi fit/selection/probe pada manifest.',
'  Ini belum merupakan audit near-duplicate atau grouping objek/sumber gambar.',
'- Verifikasi byte citra saat ini tidak diulang di sini; runner memeriksanya',
'  sebelum tiap run dan pemeriksaan kesiapan 16 Oktober telah memverifikasinya.',
'','HASIL PROBE — MEAN +/- SAMPLE SD ANTAR DUA SEED',
'Catatan: SD dari dua seeds bersifat deskriptif, bukan confidence interval.']
for ds in ['warpc','trashnet']:
 lines+=['',ds.upper(),'Konfigurasi | Macro-F1 (%) | Delta vs Base (pp), per seed | P | MMAC']
 for r in [r for r in summary if r['dataset']==ds]:
  lines.append(f"{r['model']} | {r['probe_f1_mean']:.3f} +/- {r['probe_f1_sd']:.3f} | {r['gain_seed1901_pp']:+.3f}, {r['gain_seed1902_pp']:+.3f} | {r['params']} | {r['macs']/1e6:.6f}")
lines+=['','INTERPRETASI ARSITEKTUR',
'1. Base vs DWSep: mean delta Base +2.131 pp pada WaRP-C dan +2.852 pp pada',
'   TrashNet, positif pada kedua seeds. Namun Base memakai sekitar 2.71x MACs.',
'   Ini tidak mengisolasi keuntungan topologi pada komputasi yang sama.',
'2. Base vs MCUNet-style: mean delta Base +0.130 pp pada WaRP-C dan -0.086 pp',
'   pada TrashNet. MCUNet-style memiliki sekitar 9.9% lebih banyak parameter;',
'   ini bukan kontrol jointly matched dan tidak menunjukkan dominasi Base.',
'3. Tanpa residual: +3.032 pp pada WaRP-C (kedua seeds positif), tetapi',
'   -2.141 pp pada TrashNet (kedua negatif). Ini kandidat interaksi task-topology',
'   yang perlu diuji kembali, bukan alasan menghapus residual pada semua task.',
'   Param/MAC Conv-Linear sama; operasi addition tidak dihitung dalam MAC ini.',
'4. Alokasi [14,30,36,52]: +0.819 pp pada WaRP-C, positif kedua seeds.',
'   TrashNet mean +0.565 pp tetapi delta +6.384 dan -5.254 pp antar seed.',
'   Belum ada redistribusi non-Base yang positif pada keempat konteks task-seed.',
'5. Lima redistribusi residual memenuhi band +/-1% P dan MAC pada kedua dataset.',
'   Peringkat tertinggi di probe adalah deskripsi retrospektif, bukan prosedur',
'   pemilihan arsitektur yang telah tervalidasi.',
'','TEMUAN KELAS YANG MATERIAL',
'WaRP-C probe support: bottle 651, canister 14, cans 56, cardboard 65, detergent 94.',
'17/18 run memiliki recall canister 0%. Satu run no-residual seed1902 memperoleh',
'1/14 benar (7.143%). Klaim peningkatan canister dari grouping legacy tidak',
'dapat dipakai. Channel redistribution sekarang belum menyelesaikan kelas ini.',
'Canister yang sangat kecil juga membuat diagnostik pasangan kelas berisik.',
'TrashNet probe support: cardboard32, glass40, metal32, paper47, plastic38, trash10.',
'Lihat classwise_probe.csv untuk semua kelas dan seluruh run.',
'','EVALUASI PROTOTIPE CONFUSION-RESPONSE',
'Prototipe dibuat sebelum pilot selesai. Seluruh allocation atau dataset',
'heldout dikeluarkan dari fit/scaling. Tidak ada random split baris pasangan.',
'Error berikut dalam poin persentase macro-F1, prediksi vs hasil retraining:']
for v in validation:lines.append(f"{v['holdout']} / {v['mode']}: MAE {v['mae_pp']:.3f}; {v['predictions']} prediksi.")
lines+=['Class-pair pada heldout dataset MAE5.480 vs architecture-only3.855.',
'Pada allocation holdout MAE4.137 vs architecture-only4.094.',
'Keempat estimator memilih kandidat identik pada keempat konteks dataset-seed:',
'TrashNet memilih Base pada kedua seeds; WaRP-C memilih [12,34,36,36].',
'Gain aktual pada WaRP-C -3.897 dan +2.202 pp. Dengan demikian, informasi',
'pasangan kelas belum memberi keuntungan keputusan pada pilot ini.',
'Angka ini adalah development validation, bukan final independent validation.',
'Pasangan kelas bukan sampel replikasi statistik yang independen.',
'','PEMERIKSAAN METODE YANG PERLU DIBERESKAN SEBELUM KONFIRMASI',
'- Hanya lima transfer nonreference; rank empat dan condition number57.79.',
'  Block2/3 berbagi width. Koefisien surrogate bukan efek kausal tiap block.',
'- Ada 110 fitur directional class-pair untuk lima block, dengan konteks task',
'  hanya dua. Banyak pasangan tidak menggantikan keragaman task/intervensi.',
'- predict_cm mempertahankan Base persis, tetapi kandidat nonreference memakai',
'  log-ratio smoothing0.5. Pada transfer mendekati nol, prediksi mengarah ke',
'  confusion tersmoothing dan tidak persis Base. Ini berpotensi membiasakan',
'  ranking terhadap Base, terutama pada kelas support kecil. Dampaknya belum',
'  dikuantifikasi. Perbaiki/ablate anchoring pada versi development terpisah;',
'  jangan menimpa versi dan hasil prototipe lama atau menganggapnya pasti',
'  penyebab hasil negatif. Ranking lama tetap disimpan sebagai hasil asli.',
'- Selection-probe gap besar pada beberapa WaRP-C run; contoh Base seed1901',
'  selection66.156 vs probe54.519. Tidak otomatis bukti leakage; konfirmasi',
'  pada split lain dibutuhkan sebelum menyimpulkan pola arsitektur stabil.',
'','KEPUTUSAN PENELITIAN DAN LANGKAH BERIKUT',
'A. Jangan memperbesar grid atau mengklaim algoritma unggul dari hasil ini.',
'B. Prioritaskan eksperimen stabilitas yang lebih kecil: Base, no-residual,',
'   [14,30,36,52], dan satu kontrol redistribusi yang ditetapkan sebelum run.',
'   Gunakan split stratified baru dan blok seed berpasangan. Kunci checkpoint',
'   selection, primary contrast dan kriteria manfaat sebelum membuka hasil.',
'   Jumlah split/seed dihitung sesuai efek relevan dan varians pilot; dua seed',
'   sekarang belum cukup stabil untuk menetapkan angka universal seperti50.',
'C. Audit contoh canister dan near-duplicate/source grouping lewat data yang',
'   tersedia terlebih dahulu. Jika nantinya menguji loss/rebalancing, pisahkan',
'   faktor tersebut dari alokasi channel agar novelty tetap bisa diatribusikan.',
'D. Perbaikan predictor bersifat development: bandingkan anchoring, reference-',
'   only/no-change, architecture-only, class-agnostic, dan class-pair pada',
'   objective serta kandidat sama. Jangan tune pada final assessment berikutnya.',
'E. Baru lanjutkan predictor ke task/budget baru jika ada sinyal prediksi yang',
'   membaik pada holdout; hitung biaya diagnostics dan calibration retraining.',
'F. Penilaian prior-art baru belum dilakukan dalam pemeriksaan ini. Hasil ini',
'   tidak menetapkan kebaruan terhadap NAS/pruning yang telah dipetakan sebelumnya.',
'','TINDAKAN SAAT INI',
'Pemeriksaan menggunakan CPU, tidak menjalankan training ACRE dan tidak mengubah',
'protokol pilot atau manuscript. Pipeline post-primary DKD11 tetap terpisah.',
'','SUMBER DAN LOKASI',
'Remote pilot: /home/kholiq/KD/ACRE_novelty_20261010',
'Remote analysis: /home/kholiq/KD/ACRE_method_development_20261010',
'Local snapshot: verified_pilot_snapshot.json; analisis asli: complete_pilot_analysis.json.',
'Script laporan ini: summarize_review.py. CSV dan grafik deskriptif disimpan bersama.']
(HERE/'HASIL_PEMERIKSAAN_ACRE_20261016.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl_cache'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
names=[r['model'] for r in summary if r['dataset']=='warpc']
fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained',sharey=True)
for ax,ds in zip(axes,['warpc','trashnet']):
 data=[r for r in summary if r['dataset']==ds]
 for y,r in enumerate(data):
  ax.plot([r['gain_seed1901_pp'],r['gain_seed1902_pp']],[y,y],color='#BDC6CD',lw=1)
  ax.scatter(r['gain_seed1901_pp'],y,color='#397A9C',marker='o',s=26,label='Seed 1901' if y==0 else None)
  ax.scatter(r['gain_seed1902_pp'],y,color='#C38B42',marker='s',s=26,label='Seed 1902' if y==0 else None)
 ax.axvline(0,color='#737D86',lw=.8,ls='--');ax.set_title('WaRP-C semantic 5' if ds=='warpc' else 'TrashNet 6');ax.set_xlabel('Probe macro-F1 difference from Base (pp)');ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(range(len(names)),[n.replace('allocation_','').replace('_',' ') for n in names],fontsize=8);axes[0].invert_yaxis();axes[1].legend(fontsize=8)
fig.savefig(HERE/'pilot_paired_differences.pdf');fig.savefig(HERE/'pilot_paired_differences.png',dpi=200);plt.close(fig)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name!='REVIEW_SHA256.json'}
(HERE/'REVIEW_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Review written:',HERE/'HASIL_PEMERIKSAAN_ACRE_20261016.txt')
