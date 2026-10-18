#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探查切片 B,为写 Bering 主脚本提供确切的列名 / 未分配表示 / join 关系。
只 import pandas+numpy,不碰 Bering,不受 GLIBCXX 影响。

请改下面两个路径,然后: python probe_B.py
把【全部】输出贴回给我。
"""
import pandas as pd
import numpy as np

# ====== 改成你的真实路径 ======
TRANSCRIPTS_B = "/data/qiuyijia/dataset/pancreas_cancer/transcripts.parquet"
ANNOTATION_B  = "/data/qiuyijia/dataset/pancreas_cancer/cell_groups.csv"  # 你那张 cell_id,group 表
# ==============================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 75)
print("【1】transcripts 表")
print("=" * 75)
# 先只读 schema,避免一次性全载入
try:
    import pyarrow.parquet as pq
    schema = pq.read_schema(TRANSCRIPTS_B)
    print("所有列名:", list(schema.names))
except Exception as e:
    print("pyarrow 读 schema 失败,改用 pandas 读前几行看列:", e)

# 读全表(只为统计;若内存吃紧可改 nrows)
df = pd.read_parquet(TRANSCRIPTS_B)
print(f"\n行数: {len(df):,}")
print(f"列名: {list(df.columns)}")
print("\n前 5 行:")
print(df.head())
print("\n各列 dtype:")
print(df.dtypes)

# 找出最可能是"细胞分配"的列(名字含 cell / segment 等)
cand_cols = [c for c in df.columns if any(k in c.lower() for k in ["cell", "segment", "assign"])]
print(f"\n疑似'细胞分配'列: {cand_cols}")

for c in cand_cols:
    s = df[c]
    print(f"\n----- 列 '{c}' -----")
    print(f"  dtype: {s.dtype}")
    print(f"  唯一值个数: {s.nunique(dropna=False)}")
    print(f"  缺失(NaN)数: {s.isna().sum():,}")
    # 取样几个值,看长什么样
    sample_vals = s.dropna().unique()[:8]
    print(f"  样本值(最多8个): {list(sample_vals)}")
    # 看是否有 0 / -1 / 'UNASSIGNED' / '' 这类未分配标记
    vc = s.value_counts(dropna=False)
    print(f"  出现次数最多的前 10 个值:")
    print(vc.head(10).to_string())
    # 专门探测常见的未分配表示
    for marker in [0, -1, "0", "-1", "", "UNASSIGNED", "unassigned", "None", "NA"]:
        try:
            cnt = (s == marker).sum()
            if cnt > 0:
                print(f"    含未分配候选标记 {marker!r}: {cnt:,} 条")
        except Exception:
            pass

# feature / 基因列
feat_cols = [c for c in df.columns if any(k in c.lower() for k in ["feature", "gene", "target", "name"])]
print(f"\n疑似'基因/feature'列: {feat_cols}")
for c in feat_cols[:3]:
    print(f"  '{c}' 样本: {list(df[c].dropna().astype(str).unique()[:6])}")

# 坐标列
coord_cols = [c for c in df.columns if any(k in c.lower() for k in ["x_", "y_", "z_", "_location", "global"]) or c.lower() in ["x","y","z"]]
print(f"\n疑似坐标列: {coord_cols}")

print("\n" + "=" * 75)
print("【2】注释表 (cell_id, group)")
print("=" * 75)
ann = pd.read_csv(ANNOTATION_B)
print(f"行数: {len(ann):,}")
print(f"列名: {list(ann.columns)}")
print("前 5 行:")
print(ann.head())
ann_id_col = ann.columns[0]
print(f"\n注释表 cell_id 列名假定为: '{ann_id_col}'")
print(f"  样本值: {list(ann[ann_id_col].astype(str).unique()[:8])}")
print(f"  唯一 cell 数: {ann[ann_id_col].nunique():,}")

print("\n" + "=" * 75)
print("【3】两张表的 cell_id 能否对上 (join 关系) —— 最关键")
print("=" * 75)
# 对每个疑似分配列,测它和注释表 cell_id 的交集
ann_ids = set(ann[ann_id_col].astype(str))
for c in cand_cols:
    tx_ids = set(df[c].dropna().astype(str).unique())
    inter = tx_ids & ann_ids
    print(f"\n  transcripts['{c}'] 唯一值 {len(tx_ids):,} 个")
    print(f"  与注释表 cell_id 交集: {len(inter):,} 个")
    if len(tx_ids) > 0:
        print(f"    -> 占 transcripts 该列的 {len(inter)/len(tx_ids)*100:.1f}%")
    if len(ann_ids) > 0:
        print(f"    -> 占注释表 cell_id 的 {len(inter)/len(ann_ids)*100:.1f}%")
    # 给几个对不上的例子,方便诊断格式差异
    only_tx = list(tx_ids - ann_ids)[:5]
    print(f"    transcripts 有、注释表没有的样本: {only_tx}")

print("\n" + "=" * 75)
print("探查完成。请把以上【全部】输出贴回。")
print("=" * 75)