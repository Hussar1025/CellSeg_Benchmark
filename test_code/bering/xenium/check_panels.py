#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断:两张 Xenium 切片的基因交集为什么只有 102?
是「真不同的 panel」,还是「同一个基因因大小写/别名/命名体系不同而没配上」?

只 import pandas,不碰 Bering/igraph,所以不受 GLIBCXX 报错影响。
用法: python check_panels.py
"""

import pandas as pd

# ---- 改成你的两张切片路径(默认已填你之前的)----
A_PATH = "/data/qiuyijia/dataset/pancreas_train/transcripts.parquet"
B_PATH = "/data/qiuyijia/dataset/pancreas_cancer/transcripts.parquet"

# 和 run_bering.py 一致的对照探针前缀(用 startswith 覆盖)
CTRL = ("NegControl", "antisense", "BLANK", "Blank", "Unassigned",
        "Deprecated", "Intergenic", "intergenic", "NegPrb", "Neg_", "GenomicControl")


def load_genes(path):
    df = pd.read_parquet(path, columns=["feature_name"])
    s = df["feature_name"].map(
        lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
    )
    s = s[~s.str.startswith(CTRL)]
    return set(s.unique())


ga, gb = load_genes(A_PATH), load_genes(B_PATH)
print("=" * 70)
print(f"A 基因数: {len(ga)}    B 基因数: {len(gb)}")
print(f"[1] 原始精确交集 (现在 run_bering 用的): {len(ga & gb)}")

# ---- 大小写 + 去空白 归一化后再求交集 ----
def norm(g):
    return g.strip().upper()

map_a = {}      # 归一化名 -> 原始名(A)
for g in ga:
    map_a.setdefault(norm(g), g)
map_b = {}
for g in gb:
    map_b.setdefault(norm(g), g)

inter_ci = set(map_a) & set(map_b)
print(f"[2] 大小写+去空白 归一化后交集: {len(inter_ci)}")
print(f"    -> 仅靠大小写/空白多救回: {len(inter_ci) - len(ga & gb)} 个")

# 被归一化救回(原始字符串不同、归一化后相同)的例子 = 大小写/空白的元凶
recovered = [(map_a[k], map_b[k]) for k in inter_ci if map_a[k] != map_b[k]]
print(f"[3] 被归一化救回的例子 (A原名, B原名), 最多 30 个:")
for a, b in recovered[:30]:
    print(f"      {a!r:20} <-> {b!r}")
if not recovered:
    print("      (无 —— 说明大小写/空白根本不是问题)")

# ---- 名字长什么样?判断 symbol vs Ensembl vs 其他 ----
print(f"[4] A 名字样本: {sorted(ga)[:15]}")
print(f"    B 名字样本: {sorted(gb)[:15]}")

# ---- 归一化之后仍然对不上的(真嫌疑:别名 or 真不同的基因)----
only_a = sorted(set(map_a) - set(map_b))
only_b = sorted(set(map_b) - set(map_a))
print(f"[5] 归一化后仍 A 独有: {len(only_a)} 个 -> {only_a[:40]}")
print(f"[6] 归一化后仍 B 独有: {len(only_b)} 个 -> {only_b[:40]}")

# 把完整清单也存盘,方便你贴给我或自己核
with open("panel_A_only.txt", "w") as f:
    f.write("\n".join(only_a))
with open("panel_B_only.txt", "w") as f:
    f.write("\n".join(only_b))
print("\n完整独有清单已写入 panel_A_only.txt / panel_B_only.txt")
print("=" * 70)