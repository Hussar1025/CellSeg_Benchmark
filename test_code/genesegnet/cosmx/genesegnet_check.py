"""
全面检查 /data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles 数据目录
"""
import os, subprocess
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles")
SEP = "═" * 60

def fmt(b):
    for u in ["B","KB","MB","GB","TB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024

# ── 1. 顶层目录结构 ────────────────────────────────────────
print(SEP)
print(f"  数据根目录: {DATA_ROOT}")
print(SEP)

if not DATA_ROOT.exists():
    print(f"✘ 路径不存在: {DATA_ROOT}")
    exit(1)

print("\n>>> 完整目录结构（3层）")
for p in sorted(DATA_ROOT.rglob("*")):
    depth = len(p.relative_to(DATA_ROOT).parts)
    if depth > 3:
        continue
    indent = "  " * (depth - 1)
    if p.is_dir():
        n = len(list(p.iterdir()))
        print(f"{indent}📁 {p.name}/  ({n} 项)")
    else:
        size = fmt(p.stat().st_size)
        print(f"{indent}📄 {p.name}  [{size}]")

# ── 2. 所有 CSV 文件逐一检查 ──────────────────────────────
print(f"\n{SEP}")
print("  所有 CSV 文件详情")
print(SEP)

csv_files = sorted(DATA_ROOT.rglob("*.csv"))
print(f"共找到 {len(csv_files)} 个 CSV 文件\n")

for f in csv_files:
    print(f"  [{f.name}]")
    print(f"  路径: {f}")
    print(f"  大小: {fmt(f.stat().st_size)}")
    try:
        # 行数
        res = subprocess.check_output(["wc", "-l", str(f)], text=True)
        nlines = int(res.strip().split()[0])
        print(f"  行数: {nlines:,}  (含表头)")
        # 列名 + 前2行
        df = pd.read_csv(f, nrows=2)
        print(f"  列名: {list(df.columns)}")
        print(f"  前2行:\n{df.to_string(index=False)}")
    except Exception as e:
        print(f"  读取失败: {e}")
    print()

# ── 3. 所有 TIF/JPG/PNG 图像文件 ─────────────────────────
print(f"\n{SEP}")
print("  图像文件")
print(SEP)

img_exts = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]
img_files = []
for ext in img_exts:
    img_files.extend(DATA_ROOT.rglob(ext))
img_files = sorted(img_files)

if img_files:
    # 按目录分组
    from itertools import groupby
    img_files_by_dir = {}
    for f in img_files:
        d = str(f.parent)
        img_files_by_dir.setdefault(d, []).append(f)

    for d, files in sorted(img_files_by_dir.items()):
        print(f"\n  📁 {d}  ({len(files)} 个文件)")
        total_size = sum(f.stat().st_size for f in files)
        print(f"     总大小: {fmt(total_size)}")
        from natsort import natsorted
        names = natsorted([f.name for f in files])
        print(f"     示例(前3): {names[:3]}")
        if len(names) > 3:
            print(f"     示例(后3): {names[-3:]}")
        # 检查第一个 tif 的尺寸
        tifs = [f for f in files if f.suffix.lower() in [".tif", ".tiff"]]
        if tifs:
            try:
                import tifffile
                sample = tifffile.imread(str(tifs[0]))
                print(f"     [样本] {tifs[0].name}: shape={sample.shape}  dtype={sample.dtype}")
            except Exception as e:
                print(f"     tif 读取失败: {e}")
else:
    print("  未找到图像文件")

# ── 4. 关键字匹配 ─────────────────────────────────────────
print(f"\n{SEP}")
print("  关键文件匹配")
print(SEP)

keywords = {
    "transcript / tx_file" : ["tx_file", "transcript", "Transcripts"],
    "metadata"             : ["metadata", "meta", "cell_metadata"],
    "CellComposite"        : ["CellComposite", "composite"],
    "expression matrix"    : ["exprMat", "expression", "counts"],
    "cell polygon"         : ["polygon", "boundary"],
}

for kind, pats in keywords.items():
    hits = []
    for pat in pats:
        hits.extend(DATA_ROOT.rglob(f"*{pat}*"))
    if hits:
        print(f"\n  ✔ {kind}")
        for h in sorted(hits)[:5]:
            print(f"     {fmt(h.stat().st_size):>10s}  {h.relative_to(DATA_ROOT)}")
    else:
        print(f"\n  ✘ {kind}  — 未找到")

# ── 5. 磁盘空间 ───────────────────────────────────────────
print(f"\n{SEP}")
print("  磁盘空间")
print(SEP)
import shutil
total, used, free = shutil.disk_usage(DATA_ROOT)
print(f"  总计: {fmt(total)}  已用: {fmt(used)}  可用: {fmt(free)}  ({free/total*100:.1f}% free)")

print(f"\n{SEP}")
print("  检查完成")
print(SEP)