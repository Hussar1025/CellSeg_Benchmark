#!/usr/bin/env python
import os, warnings
import numpy as np
import pandas as pd
import tables
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

DATA="yourpath"
BASE="yourpath"
D="yourpath"
COORD=D+"/yourpath"
H5=D+"/yourpath"
OUT="yourpath"

PS=0.2125
GEM_BIN=2
UNIT=PS*GEM_BIN
X0=22001*PS
Y0=8664*PS
X1=X0+10000*PS
Y1=Y0+10000*PS

CTRL=("NegControlProbe","NegControlCodeword","negcontrol",
      "antisense","BLANK","Blank","blank",
      "UnassignedCodeword","DeprecatedCodeword",
      "Intergenic","Genomic","Unassigned")

def rnd(x,n=4):
    return round(float(x),n) if np.isfinite(x) else np.nan

print("Loading GT...",flush=True)
gt=pd.read_parquet(DATA+"/cells.parquet",
                   columns=["cell_id","x_centroid","y_centroid"])
gt=gt[(gt.x_centroid>=X0)&(gt.x_centroid<X1)&
      (gt.y_centroid>=Y0)&(gt.y_centroid<Y1)].reset_index(drop=True)
gt["cell_id"]=gt["cell_id"].astype(str)

if len(gt)!=33002:
    raise RuntimeError(f"GT mismatch: {len(gt)} != 33002")

print("Loading transcripts...",flush=True)
tx=pd.read_parquet(DATA+"/transcripts.parquet",
                   columns=["cell_id","feature_name",
                            "x_location","y_location","qv"])
tx=tx[(tx.qv>=20)&
      (tx.x_location>=X0)&(tx.x_location<X1)&
      (tx.y_location>=Y0)&(tx.y_location<Y1)].copy()
tx["feature_name"]=tx["feature_name"].astype(str)
tx=tx[~tx.feature_name.str.startswith(CTRL,na=False)].reset_index(drop=True)
tx["cell_id"]=tx["cell_id"].astype(str)
print("GT =",len(gt)," transcripts =",len(tx),flush=True)

print("Loading Cellist coordinates...",flush=True)
c=pd.read_csv(COORD,sep="\t")
pred=pd.DataFrame({
    "cell":c["Cellist"].astype(float),
    "x_um":c["x"].astype(float)*UNIT+X0,
    "y_um":c["y"].astype(float)*UNIT+Y0
}).drop_duplicates("cell").reset_index(drop=True)

print("Pred =",len(pred),flush=True)

gxy=gt[["x_centroid","y_centroid"]].to_numpy(float)
pxy=pred[["x_um","y_um"]].to_numpy(float)

# match radius: GT NN p95 * 1.25
tree_g=cKDTree(gxy)
nn,_=tree_g.query(gxy,k=2)
R=max(2.0,float(np.quantile(nn[:,1],0.95)*1.25))
AR=R*1.5
print(f"match radius = {R:.4f} um",flush=True)

# nearest-GT matching, one best prediction per GT
dd,gi=tree_g.query(pxy,k=1,distance_upper_bound=R)
best={}
for pi in range(len(pred)):
    if not np.isfinite(dd[pi]) or dd[pi]>=R:
        continue
    g=int(gi[pi])
    if g not in best or dd[pi]<best[g][1]:
        best[g]=(pi,float(dd[pi]))
pairs=[(pi,g,d) for g,(pi,d) in best.items()]
nmatch=len(pairs)
sh=np.array([z[2] for z in pairs],float)

prec=nmatch/len(pred)
rec=nmatch/len(gt)
f1=2*prec*rec/(prec+rec)

print(f"matched={nmatch} precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}",flush=True)

# transcript -> nearest predicted centroid
txy=tx[["x_location","y_location"]].to_numpy(float)
td,tpi=cKDTree(pxy).query(txy,k=1,distance_upper_bound=AR)
ok=np.isfinite(td)&(td<AR)
tx["pred_row"]=np.where(ok,tpi,-1)

# counts
pred_cnt=tx.loc[tx.pred_row>=0,"pred_row"].value_counts()
valid_gt=~tx["cell_id"].isin(["UNASSIGNED","nan","None",""])
gt_cnt=tx.loc[valid_gt,"cell_id"].value_counts()

pc=np.array([pred_cnt.get(p[0],0) for p in pairs],float)
gc=np.array([gt_cnt.get(gt.cell_id.iloc[p[1]],0) for p in pairs],float)
m=(pc>0)&(gc>0)

cp=pearsonr(pc[m],gc[m])[0] if m.sum()>10 else np.nan
cs=spearmanr(pc[m],gc[m])[0] if m.sum()>10 else np.nan
mae=np.mean(np.abs(pc[m]-gc[m])) if m.sum()>10 else np.nan
rmse=np.sqrt(np.mean((pc[m]-gc[m])**2)) if m.sum()>10 else np.nan

# Cellist expression vectors; preserve barcode-ID mapping
print("Loading Cellist expression...",flush=True)
with tables.open_file(H5,"r") as h:
    z=h.root.matrix
    mat=csc_matrix((z.data[:],z.indices[:],z.indptr[:]),
                   shape=tuple(z.shape[:]))
    bcs=[x.decode() if isinstance(x,bytes) else str(x)
         for x in z.barcodes[:]]
    genes=[x.decode() if isinstance(x,bytes) else str(x)
           for x in z.features.name[:]]

id2col={}
for j,b in enumerate(bcs):
    try: id2col[float(b)]=j
    except: pass

sub=tx[valid_gt]
gtp=sub.groupby(["cell_id","feature_name"]).size().unstack(fill_value=0)
common=[g for g in genes if g in gtp.columns]
gcol=[genes.index(g) for g in common]
gtp=gtp[common]

rng=np.random.default_rng(0)
samp=pairs if len(pairs)<=3000 else [
    pairs[i] for i in rng.choice(len(pairs),3000,replace=False)
]

cosl,jsl,prl=[],[],[]
for pi,gj,_ in samp:
    col=id2col.get(float(pred.cell.iloc[pi]))
    gid=gt.cell_id.iloc[gj]
    if col is None or gid not in gtp.index:
        continue
    a=np.asarray(mat[:,col].todense()).ravel()[gcol].astype(float)
    b=gtp.loc[gid].to_numpy(float)
    if a.sum()<5 or b.sum()<5:
        continue
    den=np.linalg.norm(a)*np.linalg.norm(b)
    if den>0:
        cosl.append(float(a@b/den))
        jsl.append(float(jensenshannon(a/a.sum(),b/b.sum())))
        if a.std()>0 and b.std()>0:
            prl.append(float(pearsonr(a,b)[0]))

vc=np.mean(cosl) if len(cosl)>10 else np.nan
vj=np.mean(jsl) if len(jsl)>10 else np.nan
vp=np.mean(prl) if len(prl)>10 else np.nan

# assignment accuracy
row2gt={p[0]:gt.cell_id.iloc[p[1]] for p in pairs}
s=tx[(tx.pred_row>=0)&valid_gt].copy()
expected=s.pred_row.map(row2gt)
inp=expected.notna()
overlap=float(inp.mean()) if len(s) else np.nan
acc=float((expected[inp].astype(str)==s.loc[inp,"cell_id"].astype(str)).mean()) if inp.any() else np.nan

res=dict(
 method="Cellist",dataset="xenium_8",
 gt_cells=len(gt),pred_cells=len(pred),
 cell_ratio=rnd(len(pred)/len(gt)),
 matched=nmatch,precision=rnd(prec),recall=rnd(rec),f1=rnd(f1),
 loc_mean_um=rnd(np.mean(sh)),loc_median_um=rnd(np.median(sh)),
 loc_p95_um=rnd(np.percentile(sh,95)),
 count_pearson=rnd(cp),count_spearman=rnd(cs),
 count_mae=rnd(mae,2),count_rmse=rnd(rmse,2),
 vec_cosine=rnd(vc),vec_js_dist=rnd(vj),vec_pearson=rnd(vp),
 vec_valid_pairs=len(cosl),
 assign_overlap=rnd(overlap),assign_accuracy=rnd(acc),
 match_radius_um=rnd(R),
 transcript_rows=len(tx)
)

os.makedirs(os.path.dirname(OUT),exist_ok=True)
pd.DataFrame([res]).to_csv(OUT,index=False)

print("\n===== FINAL =====")
for k,v in res.items(): print(f"{k:22} {v}")
print("saved ->",OUT)