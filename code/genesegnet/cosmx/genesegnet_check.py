"""
Full inspection of the yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles data directory
"""
import os, subprocess
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles")
SEP = "═" * 60

def fmt(b):
    for u in ["B","KB","MB","GB","TB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024

# -- 1. Top-level directory structure ------------------------
print(SEP)
print(f"  data root: {DATA_ROOT}")
print(SEP)

if not DATA_ROOT.exists():
    print(f"✘ path does not exist: {DATA_ROOT}")
    exit(1)

print("\n>>> Full directory structure (3 levels)")
for p in sorted(DATA_ROOT.rglob("*")):
    depth = len(p.relative_to(DATA_ROOT).parts)
    if depth > 3:
        continue
    indent = "  " * (depth - 1)
    if p.is_dir():
        n = len(list(p.iterdir()))
        print(f"{indent}📁 {p.name}/  ({n} entries)")
    else:
        size = fmt(p.stat().st_size)
        print(f"{indent}📄 {p.name}  [{size}]")

# -- 2. Check every CSV file one by one ----------------------
print(f"\n{SEP}")
print("  all CSV file details")
print(SEP)

csv_files = sorted(DATA_ROOT.rglob("*.csv"))
print(f"found {len(csv_files)} CSV files\n")

for f in csv_files:
    print(f"  [{f.name}]")
    print(f"  path: {f}")
    print(f"  size: {fmt(f.stat().st_size)}")
    try:
        # line count
        res = subprocess.check_output(["wc", "-l", str(f)], text=True)
        nlines = int(res.strip().split()[0])
        print(f"  lines: {nlines:,}  (incl. header)")
        # column names + first 2 rows
        df = pd.read_csv(f, nrows=2)
        print(f"  columns: {list(df.columns)}")
        print(f"  first 2 rows:\n{df.to_string(index=False)}")
    except Exception as e:
        print(f"  read failed: {e}")
    print()

# -- 3. All TIF/JPG/PNG image files -------------------------
print(f"\n{SEP}")
print("  image files")
print(SEP)

img_exts = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]
img_files = []
for ext in img_exts:
    img_files.extend(DATA_ROOT.rglob(ext))
img_files = sorted(img_files)

if img_files:
    # group by directory
    from itertools import groupby
    img_files_by_dir = {}
    for f in img_files:
        d = str(f.parent)
        img_files_by_dir.setdefault(d, []).append(f)

    for d, files in sorted(img_files_by_dir.items()):
        print(f"\n  📁 {d}  ({len(files)} files)")
        total_size = sum(f.stat().st_size for f in files)
        print(f"     total size: {fmt(total_size)}")
        from natsort import natsorted
        names = natsorted([f.name for f in files])
        print(f"     examples (first 3): {names[:3]}")
        if len(names) > 3:
            print(f"     examples (last 3): {names[-3:]}")
        # check dimensions of the first tif
        tifs = [f for f in files if f.suffix.lower() in [".tif", ".tiff"]]
        if tifs:
            try:
                import tifffile
                sample = tifffile.imread(str(tifs[0]))
                print(f"     [sample] {tifs[0].name}: shape={sample.shape}  dtype={sample.dtype}")
            except Exception as e:
                print(f"     tif read failed: {e}")
else:
    print("  no image files found")

# -- 4. Keyword matching ------------------------------------
print(f"\n{SEP}")
print("  key file matching")
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
        print(f"\n  ✘ {kind}  -- not found")

# -- 5. Disk space ------------------------------------------
print(f"\n{SEP}")
print("  disk space")
print(SEP)
import shutil
total, used, free = shutil.disk_usage(DATA_ROOT)
print(f"  total: {fmt(total)}  used: {fmt(used)}  free: {fmt(free)}  ({free/total*100:.1f}% free)")

print(f"\n{SEP}")
print("  check done")
print(SEP)