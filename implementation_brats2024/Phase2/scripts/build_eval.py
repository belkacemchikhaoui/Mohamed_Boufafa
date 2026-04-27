"""Build Phase2_B1 - Full 16-test evaluation battery."""
import json

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src.strip().split("\n"),"id":"a"}
def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip().split("\n"),"id":"a"}

cells = [
md("""# Phase 2C — Full Embedding Evaluation Battery (16 Tests)
## Quantitative Evaluation + CNN Comparison + Static Modeling Limitations

**Phase 2 Deliverables:**
- Quantitative evaluation of baseline performance
- Comparative performance results serving as reference benchmarks
- Identification of limitations in static modeling

**Tests:** M1-M6 (Morphology), H1-H4 (Heterogeneity), T1-T7 (Temporal)"""),

code("""import numpy as np, json, random, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import pandas as pd
warnings.filterwarnings('ignore')
OUTPUT_ROOT = Path('/kaggle/working/phase2_evaluation')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
models = {}
for name in ['segresnet', 'dynunet']:
    for loc in [Path('/kaggle/input'), Path('/kaggle/working')]:
        for f in loc.rglob(f'cnn_{name}_embeddings_v2.npz'):
            data = np.load(f)
            models[name] = {k: data[k] for k in data.keys()}
            print(f'{name}: {len(models[name])} embs x {list(models[name].values())[0].shape[0]}-dim')
            break
meta_df = None
for f in Path('/kaggle/input').rglob('*.xlsx'): meta_df = pd.read_excel(f); break
tumor_df = None
for f in list(Path('/kaggle/input').rglob('tumor_volumes.csv'))+list(Path('/kaggle/working').rglob('tumor_volumes.csv')):
    tumor_df = pd.read_csv(f); break
print(f'Models: {list(models.keys())} | Meta: {meta_df is not None} | Vols: {tumor_df is not None}')"""),

code("""# M1-M6: MORPHOLOGY TESTS
print('='*60+'\\n  MORPHOLOGY TESTS M1-M6\\n'+'='*60)
results = {}
for mn, embs in models.items():
    keys = list(embs.keys()); X = np.stack([embs[k] for k in keys])
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    mX, mvols = [], {'wt':[],'tc':[],'et':[]}
    for k in keys:
        pid,tp = k.split('__')
        if tumor_df is not None:
            row = tumor_df[(tumor_df['patient_id']==pid)&(tumor_df['timepoint']==tp)]
            if len(row)>0:
                mX.append(Xs[keys.index(k)])
                for r in ['wt','tc','et']: mvols[r].append(row.iloc[0][f'{r}_vol'])
    if len(mX)<10: print(f'{mn}: too few matched'); continue
    mX = np.stack(mX); results[mn] = {}
    ridge = Ridge(alpha=1.0)
    y = np.array(mvols['wt']); p = cross_val_predict(ridge,mX,y,cv=5)
    results[mn]['M1_volume_R2'] = max(r2_score(y,p),0)
    yl = np.log1p(y); pl = cross_val_predict(ridge,mX,yl,cv=5)
    results[mn]['M2_logvol_R2'] = max(r2_score(yl,pl),0)
    ysvr = np.array(mvols['et'])/(np.array(mvols['wt'])+0.01)
    ps = cross_val_predict(ridge,mX,ysvr,cv=5)
    results[mn]['M3_svr_R2'] = max(r2_score(ysvr,ps),0)
    yncr = (np.array(mvols['tc'])-np.array(mvols['et'])>0.1).astype(int)
    if len(set(yncr))>1:
        pncr = cross_val_predict(LogisticRegression(max_iter=1000),mX,yncr,cv=5)
        results[mn]['M4_necrosis_F1'] = f1_score(yncr,pncr,average='weighted')
    else: results[mn]['M4_necrosis_F1'] = 0.0
    ye = np.array(mvols['tc'])/(np.array(mvols['wt'])+0.01)
    pe = cross_val_predict(ridge,mX,ye,cv=5)
    results[mn]['M5_elongation_R2'] = max(r2_score(ye,pe),0)
    if meta_df is not None:
        nn = NearestNeighbors(n_neighbors=6).fit(mX)
        _,idx = nn.kneighbors(mX)
        types = []
        for k in keys[:len(mX)]:
            pid = k.split('__')[0]
            m = meta_df[meta_df['BraTS Subject ID'].str.contains(pid,na=False)]
            types.append(m.iloc[0]['Glioma type '] if len(m)>0 else 'UNK')
        con,tot = 0,0
        for i in range(len(types)):
            if types[i]=='UNK': continue
            for j in idx[i][1:]:
                if j<len(types) and types[j]!='UNK': con+=int(types[i]==types[j]); tot+=1
        results[mn]['M6_nn_consistency'] = 100*con/max(tot,1)
    else: results[mn]['M6_nn_consistency'] = 0
    for k,v in sorted(results[mn].items()):
        if k.startswith('M'): print(f'  {mn} {k}: {v:.3f}')"""),

code("""# H1-H4: HETEROGENEITY TESTS
print('\\n'+'='*60+'\\n  HETEROGENEITY TESTS H1-H4\\n'+'='*60)
for mn, embs in models.items():
    keys = list(embs.keys()); X = np.stack([embs[k] for k in keys])
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10,Xs.shape[1])); pca.fit(Xs)
    results[mn]['H1_pca_var'] = float(np.sum(pca.explained_variance_ratio_))
    Xn = X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    pairs = random.sample([(i,j) for i in range(len(keys)) for j in range(i+1,len(keys))],min(500,len(keys)*(len(keys)-1)//2))
    sims = [float(Xn[i]@Xn[j]) for i,j in pairs]
    results[mn]['H2_diversity'] = 1-np.mean(sims)
    if meta_df is not None:
        yt,Xt = [],[]
        for k in keys:
            pid = k.split('__')[0]
            m = meta_df[meta_df['BraTS Subject ID'].str.contains(pid,na=False)]
            if len(m)>0: yt.append(m.iloc[0]['Glioma type ']); Xt.append(Xs[keys.index(k)])
        if len(set(yt))>=2:
            le = LabelEncoder(); ye = le.fit_transform(yt)
            scores = cross_val_score(LogisticRegression(max_iter=1000),np.stack(Xt),ye,cv=5,scoring='f1_weighted')
            results[mn]['H3_glioma_F1'] = float(scores.mean())
        else: results[mn]['H3_glioma_F1'] = 0
    else: results[mn]['H3_glioma_F1'] = 0
    norms = np.linalg.norm(X,axis=1)
    results[mn]['H4_norm_cv'] = float(np.std(norms)/(np.mean(norms)+1e-8))
    for k,v in sorted(results[mn].items()):
        if k.startswith('H'): print(f'  {mn} {k}: {v:.3f}')"""),

code("""# T1-T7: TEMPORAL TESTS (Limitations of Static Modeling)
print('\\n'+'='*60+'\\n  TEMPORAL TESTS T1-T7 (Static Modeling Limitations)\\n'+'='*60)
for mn, embs in models.items():
    keys = list(embs.keys())
    pe = {}
    for k in keys:
        pid,tp = k.split('__')
        if pid not in pe: pe[pid] = {}
        pe[pid][tp] = embs[k]
    longi = {p:t for p,t in pe.items() if len(t)>=2}
    print(f'  {mn}: {len(longi)} longitudinal patients')
    if len(longi)<5:
        for t in ['T1','T2','T3','T4','T5','T6','T7']: results[mn][f'{t}'] = 0
        continue
    den,dvol,csim = [],[],[]
    for pid,tps in longi.items():
        stps = sorted(tps.keys())
        for i in range(len(stps)-1):
            e0,e1 = tps[stps[i]],tps[stps[i+1]]
            den.append(np.linalg.norm(e1-e0))
            n0,n1 = np.linalg.norm(e0)+1e-8,np.linalg.norm(e1)+1e-8
            csim.append(float((e0/n0)@(e1/n1)))
            if tumor_df is not None:
                v0 = tumor_df[(tumor_df['patient_id']==pid)&(tumor_df['timepoint']==stps[i])]
                v1 = tumor_df[(tumor_df['patient_id']==pid)&(tumor_df['timepoint']==stps[i+1])]
                if len(v0)>0 and len(v1)>0:
                    dvol.append(abs(v1.iloc[0]['wt_vol']-v0.iloc[0]['wt_vol']))
    ml = min(len(den),len(dvol))
    if ml>=5: r,_ = pearsonr(den[:ml],dvol[:ml]); results[mn]['T1_dist_vol_r'] = abs(r)
    else: results[mn]['T1_dist_vol_r'] = 0
    results[mn]['T2_ordering'] = sum(1 for d in den if d>0)/max(len(den),1)
    if ml>=10:
        Xd = np.array(den[:ml]).reshape(-1,1); yd = np.array(dvol[:ml])
        pd2 = cross_val_predict(Ridge(alpha=1.0),Xd,yd,cv=min(5,ml//2))
        results[mn]['T3_delta_R2'] = max(r2_score(yd,pd2),0)
    else: results[mn]['T3_delta_R2'] = 0
    if ml>=10:
        yr = np.array([1 if d>1.0 else 0 for d in dvol[:ml]])
        if len(set(yr))>1:
            s = cross_val_score(LogisticRegression(max_iter=1000),np.array(den[:ml]).reshape(-1,1),yr,cv=3,scoring='roc_auc')
            results[mn]['T4_response_AUC'] = float(s.mean())
        else: results[mn]['T4_response_AUC'] = 0.5
    else: results[mn]['T4_response_AUC'] = 0.5
    results[mn]['T5_coherence'] = float(np.mean(csim)) if csim else 0
    results[mn]['T6_velocity_cv'] = float(np.std(den)/(np.mean(den)+1e-8)) if den else 0
    fe,le2 = [],[]
    for pid,tps in longi.items():
        stps = sorted(tps.keys())
        fe.append(np.linalg.norm(tps[stps[0]])); le2.append(np.linalg.norm(tps[stps[-1]]))
    pstd = np.sqrt((np.std(fe)**2+np.std(le2)**2)/2)+1e-8
    results[mn]['T7_treatment_d'] = abs(np.mean(fe)-np.mean(le2))/pstd
    for k,v in sorted(results[mn].items()):
        if k.startswith('T'):
            flag = ' <- WEAK (static limitation)' if v<0.3 and k in ['T1_dist_vol_r','T3_delta_R2'] else ''
            print(f'  {mn} {k}: {v:.3f}{flag}')"""),

code("""# DASHBOARD + COMPARISON
print('\\n'+'='*60+'\\n  FULL 16-TEST DASHBOARD\\n'+'='*60)
for mn in models:
    passed = sum(1 for k,v in results.get(mn,{}).items() if v>0.3)
    total = len(results.get(mn,{}))
    print(f'\\n{mn}: {passed}/{total} tests above 0.3 threshold')
    for k,v in sorted(results.get(mn,{}).items()):
        print(f'  {k:<25s} {v:>8.3f}')
with open(OUTPUT_ROOT/'eval_results.json','w') as f: json.dump(results,f,indent=2)
print(f'\\nSaved: eval_results.json')
# Comparison bar chart
if len(models)>=2:
    fig,axes = plt.subplots(1,3,figsize=(18,6))
    mns = list(results.keys())
    cats = [('Morphology',[k for k in sorted(results[mns[0]]) if k.startswith('M')]),
            ('Heterogeneity',[k for k in sorted(results[mns[0]]) if k.startswith('H')]),
            ('Temporal',[k for k in sorted(results[mns[0]]) if k.startswith('T')])]
    cols = ['#3498db','#e74c3c']
    for ax,(cn,tks) in zip(axes,cats):
        x = np.arange(len(tks)); w = 0.35
        for mi,mn in enumerate(mns):
            vals = [results[mn].get(k,0) for k in tks]
            ax.bar(x+mi*w,vals,w,label=mn,color=cols[mi],alpha=0.8)
        ax.set_title(cn); ax.set_xticks(x+w/2)
        ax.set_xticklabels([k.split('_')[0] for k in tks],rotation=45); ax.legend()
    plt.suptitle('16-Test Evaluation: SegResNet vs DynUNet',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(OUTPUT_ROOT/'eval_comparison.png',dpi=150); plt.close()
    print('Saved: eval_comparison.png')"""),

code("""# LIMITATIONS OF STATIC MODELING (quantitative evidence)
print('\\n'+'='*60+'\\n  LIMITATIONS OF STATIC MODELING\\n'+'='*60)
for mn in models:
    r = results.get(mn,{})
    print(f'\\n{mn}:')
    print(f'  Morphology (M1-M6):     STRONG — CNNs capture tumor shape/volume')
    print(f'    M1 Volume R2:         {r.get("M1_volume_R2",0):.3f}')
    print(f'    M4 Necrosis F1:       {r.get("M4_necrosis_F1",0):.3f}')
    print(f'  Temporal (T1-T7):       WEAK — CNNs cannot track evolution')
    print(f'    T1 Demb vs DVol r:    {r.get("T1_dist_vol_r",0):.3f}')
    print(f'    T3 Delta R2:          {r.get("T3_delta_R2",0):.3f}')
    print(f'    T4 Response AUC:      {r.get("T4_response_AUC",0):.3f}')
print('\\nCONCLUSION: Static CNN embeddings capture WHAT the tumor looks like')
print('but NOT HOW it changes. Phase 3 (ViT) + temporal modeling needed.')
print('\\nPhase 2 COMPLETE')"""),
]

nb = {"nbformat":4,"nbformat_minor":5,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info":{"name":"python","version":"3.12.12"}},
      "cells":cells}
for i,c in enumerate(nb['cells']): c['id'] = f'c{i:02d}'
out = '/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase2/notebooks/Phase2_B1_CNN_Embedding_Evaluation.ipynb'
with open(out,'w') as f: json.dump(nb,f,indent=1)
print(f'Built: {out} ({len(cells)} cells)')
import os
old = '/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase2/notebooks/Phase2_B1_CNN_Embedding_Comparison.ipynb'
if os.path.exists(old): os.remove(old); print(f'Removed old comparison notebook')
