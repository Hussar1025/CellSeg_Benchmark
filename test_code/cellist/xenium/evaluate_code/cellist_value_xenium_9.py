#!/usr/bin/env python
import os, warnings, json
import tifffile
import numpy as np
import pandas as pd
import tables
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

DATA="/data/qiuyijia/dataset/xenium9"
BASE="/data/qiuyijia/cellist_xenium_xenium_9_roi10000"
D=BASE+"/cellist/alpha_0.8_sigma_1.0_beta_10_gene_HVG_dist_18_twostep_False_cyto_False_noise_0.0_neigh_2.5"
COORD=D+"/xenium_9_Cellist_segmentation_cell_coord.txt"
H5=D+"/xenium_9_Cellist_segmentation_cell_count.h5"
OUT="/data/qiuyijia/eval_results_qv20/xenium_9/cellist_xenium_9_metrics.csv"

PS=0.2125
GEM_BIN=2
UNIT=PS*GEM_BIN
X0=16374*PS
Y0=5147*PS
X1=X0+10000*PS
Y1=Y0+10000*PS

CTRL=("NegControlProbe","NegControlCodeword","negcontrol",
      "antisense","BLANK","Blank","blank",
      "UnassignedCodeword","DeprecatedCodeword",
      "Intergenic","Genomic","Unassigned")

def rnd(x,n=4):
    return round(float(x),n) if np.isfinite(x) else np.nan

print("Loading GT...",flush=True)
gt=pd.read_parquet(
    "/data/qiuyijia/dataset/xenium9/benchmark_roi_10000/cells.parquet",
    columns=["cell_id","x_centroid","y_centroid"]
)
# xenium9 benchmark_roi_10000/cells.parquet is already ROI-local in um.
# Convert the complete locked GT to the same global-um frame used by Cellist predictions.
gt = gt.reset_index(drop=True)
gt["x_centroid"] = gt["x_centroid"].astype(float) + X0
gt["y_centroid"] = gt["y_centroid"].astype(float) + Y0
gt["cell_id"]=gt["cell_id"].astype(str)

if len(gt)!=17689:
    raise RuntimeError(f"GT mismatch: {len(gt)} != 17689")

print("Loading transcripts...",flush=True)
TX_CACHE="/data/qiuyijia/dataset/xenium9/_roi_cache/tx_roi_5147_16374_10000_qv20.parquet"
tx=pd.read_parquet(TX_CACHE)

# x9 ROI cache is already qv20-filtered and ROI-filtered.
# Harmonise its schema with the existing Cellist evaluator convention.
tx=tx.rename(columns={
    "gene": "feature_name",
    "platform_cell": "cell_id",
    "x_um": "x_location",
    "y_um": "y_location",
})
tx["feature_name"]=tx["feature_name"].astype(str)
tx=tx[~tx.feature_name.str.startswith(CTRL,na=False)].reset_index(drop=True)
tx["cell_id"]=tx["cell_id"].fillna("").astype(str)
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

# same x9 convention: GT NN p95 * 1.25
tree_g=cKDTree(gxy)
nn,_=tree_g.query(gxy,k=2)
R=max(2.0,float(np.quantile(nn[:,1],0.95)*1.25))
AR=R*1.5
print(f"match radius = {R:.4f} um",flush=True)

# same Cellist convention: nearest-GT, one best prediction per GT
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

# transcript -> nearest predicted centroid, same x2 Cellist evaluator convention
txy=tx[["x_location","y_location"]].to_numpy(float)
td,tpi=cKDTree(pxy).query(txy,k=1,distance_upper_bound=AR)
ok=np.isfinite(td)&(td<AR)
tx["pred_row"]=np.where(ok,tpi,-1)

# canonical GT transcript assignment: same x9 benchmark truth used by UCS
PRIOR="/data/qiuyijia/SpaSegBench/result/ucs/xenium_9/platform_cell_labels.tif"
PRIOR_IDS="/data/qiuyijia/SpaSegBench/result/ucs/xenium_9/platform_cell_ids.json"

prior=np.squeeze(tifffile.imread(PRIOR))
name2lab=json.load(open(PRIOR_IDS))
lab2name={int(v):str(k) for k,v in name2lab.items()}

# tx coordinates in cache are global um; prior raster is ROI-local native pixels
cc0=np.floor((tx["x_location"].to_numpy(float)-X0)/PS).astype(np.int64)
rr0=np.floor((tx["y_location"].to_numpy(float)-Y0)/PS).astype(np.int64)
inside=((rr0>=0)&(rr0<prior.shape[0])&
        (cc0>=0)&(cc0<prior.shape[1]))

raw_gt=np.zeros(len(tx),dtype=np.int64)
raw_gt[inside]=prior[rr0[inside],cc0[inside]].astype(np.int64)

gt_lab=np.full(len(tx),"",dtype=object)
nz=raw_gt>0
gt_lab[nz]=[lab2name.get(int(v),"") for v in raw_gt[nz]]
tx["gt_cell_id"]=gt_lab

valid_gt=tx["gt_cell_id"]!=""

print("canonical GT assignment =",
      f"{valid_gt.mean()*100:.2f}% "
      f"({valid_gt.sum()}/{len(tx)})", flush=True)

# counts
pred_cnt=tx.loc[tx.pred_row>=0,"pred_row"].value_counts()
gt_cnt=tx.loc[valid_gt,"gt_cell_id"].value_counts()

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

sub=tx.loc[valid_gt,["gt_cell_id","feature_name"]]
gtp=sub.groupby(["gt_cell_id","feature_name"]).size().unstack(fill_value=0)
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

# assignment accuracy via matched Cellist <-> canonical GT cell pairs
row2gt={p[0]:gt.cell_id.iloc[p[1]] for p in pairs}
s=tx[(tx.pred_row>=0)&valid_gt].copy()
expected=s.pred_row.map(row2gt)
inp=expected.notna()
overlap=float(inp.mean()) if len(s) else np.nan
acc=float(
    (expected[inp].astype(str)==s.loc[inp,"gt_cell_id"].astype(str)).mean()
) if inp.any() else np.nan

res=dict(
 method="Cellist",dataset="xenium_9",
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
