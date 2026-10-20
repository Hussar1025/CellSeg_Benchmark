#!/usr/bin/env bash
# =============================================================================
# Rebuild the BOMS C++ extension from scratch
# =============================================================================
# Background: the precompiled boms_wrapper.*.so segfaults on this machine
# (synthetic data collapses into a single point + Segmentation fault), i.e.
# binary incompatible. The old server is gone, so it must be rebuilt locally.
#
# This script does four things:
#   1. Probe the source location and build system (Cython? pybind11? cppimport?)
#   2. Check that the compiler / OpenMP / numpy headers are present
#   3. Compile from scratch
#   4. Verify with synthetic data (3 point clusters must yield >=3 modes,
#      not collapse into 1)
#
# Usage:
#   conda activate bboms
#   bash rebuild_boms.sh            # fully automatic
#   bash rebuild_boms.sh --probe    # probe only, no compile
# =============================================================================
set -o pipefail

PROBE_ONLY=0
[ "$1" = "--probe" ] && PROBE_ONLY=1

SEP="========================================================================"
PKG_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/boms"
SRC_GUESS_1="yourpath/boms"
SRC_GUESS_2="yourpath/boms/code"

echo "$SEP"
echo "[1] probe source and build system"
echo "$SEP"
echo "  CONDA_PREFIX = $CONDA_PREFIX"
echo "  installed boms package = $PKG_DIR"
echo

# C++ sources inside the installed package
echo "  C++/pyx source files in the installed package:"
find "$PKG_DIR" -maxdepth 2 \( -name "*.cpp" -o -name "*.pyx" -o -name "*.h" \
     -o -name "*.hpp" -o -name "*.cc" \) 2>/dev/null | while read f; do
    echo "    $f  [$(du -h "$f" | cut -f1)]"
done

# find the source repo (setup.py / pyproject.toml)
echo
echo "  search for source repos (containing setup.py or meanshift source):"
for d in "$SRC_GUESS_1" "$SRC_GUESS_2" "$HOME/boms" ; do
    [ -d "$d" ] || continue
    for marker in setup.py pyproject.toml meson.build; do
        if [ -f "$d/$marker" ]; then
            echo "    ✓ $d/$marker"
        fi
    done
    # find the meanshift C++ source
    find "$d" -maxdepth 3 \( -name "*meanshift*" -o -name "*wrapper*" \) \
         \( -name "*.cpp" -o -name "*.pyx" -o -name "*.h" -o -name "*.hpp" \) \
         2>/dev/null | head -10 | while read f; do
        echo "      source: $f"
    done
done

echo
echo "$SEP"
echo "[2] check the build toolchain"
echo "$SEP"
echo -n "  gcc      : "; gcc --version 2>/dev/null | head -1 || echo "✘ missing"
echo -n "  g++      : "; g++ --version 2>/dev/null | head -1 || echo "✘ missing"
echo -n "  cython   : "; python -c "import Cython; print(Cython.__version__)" 2>/dev/null || echo "✘ missing (pip install cython)"
echo -n "  pybind11 : "; python -c "import pybind11; print(pybind11.__version__)" 2>/dev/null || echo "(not installed, may not be needed)"
echo -n "  numpy hdr: "; python -c "import numpy; print(numpy.get_include())" 2>/dev/null

# OpenMP header + library
echo -n "  omp.h    : "
echo '#include <omp.h>
int main(){return omp_get_num_threads();}' > /tmp/_omptest.c
if gcc -fopenmp /tmp/_omptest.c -o /tmp/_omptest 2>/tmp/_omperr; then
    echo "✓ OpenMP compiles"
else
    echo "✘ OpenMP compile failed:"; cat /tmp/_omperr
fi

echo -n "  libgomp  : "; ls "$CONDA_PREFIX"/lib/libgomp.so* 2>/dev/null | head -1 || echo "system path"

if [ "$PROBE_ONLY" = "1" ]; then
    echo
    echo "  --probe mode, stopping here. Paste the output above back to decide the compile command."
    exit 0
fi

echo
echo "$SEP"
echo "[3] compile"
echo "$SEP"

# prefer: if a source repo is found, reinstall via its setup.py
BUILT=0
for d in "$SRC_GUESS_1" "$SRC_GUESS_2" ; do
    if [ -f "$d/setup.py" ] || [ -f "$d/pyproject.toml" ]; then
        echo "  -> reinstall from source repo: $d"
        echo "    first uninstall the old precompiled package..."
        pip uninstall -y boms 2>/dev/null
        cd "$d"
        # force a recompile, don't reuse the cached wheel
        echo "    pip install -e . --no-build-isolation --force-reinstall ..."
        if CC=gcc CXX=g++ CFLAGS="-fopenmp -O3" LDFLAGS="-fopenmp" \
           pip install -e . --no-build-isolation --force-reinstall 2>&1 | tail -40; then
            BUILT=1
            break
        fi
    fi
done

# fallback: pull the source from GitHub
if [ "$BUILT" = "0" ]; then
    echo
    echo "  no compilable source repo found locally, pulling from GitHub..."
    CLONE_DIR="yourpath/boms/boms_src_$(date +%s)"
    if git clone https://github.com/sciai-lab/boms "$CLONE_DIR" 2>&1 | tail -5; then
        cd "$CLONE_DIR"
        pip uninstall -y boms 2>/dev/null
        echo "    build and install from $CLONE_DIR..."
        CC=gcc CXX=g++ CFLAGS="-fopenmp -O3" LDFLAGS="-fopenmp" \
           pip install -e . --no-build-isolation 2>&1 | tail -40
        BUILT=1
    else
        echo "  ✘ git clone failed — check network access to github.com"
        echo "    (if network is restricted, ask an admin to allow github.com)"
        exit 1
    fi
fi

echo
echo "$SEP"
echo "[4] verify: 3 point clusters must yield >=3 modes"
echo "$SEP"
python - <<'PYEOF'
import numpy as np
try:
    import boms
    import boms_wrapper as ms
except Exception as e:
    print(f"  ✘ import failed: {e}")
    raise SystemExit(1)

rng = np.random.default_rng(0)
xs, ys, gs = [], [], []
for k, (cx, cy) in enumerate([(20,20),(60,20),(40,60)]):
    xs.append(rng.normal(cx,3,300)); ys.append(rng.normal(cy,3,300))
    gs.append(np.full(300,k))
x = np.concatenate(xs); y = np.concatenate(ys)
g = np.concatenate(gs).astype("int32")[:,None]
coords = np.stack([x,y],1).astype("float32")

out = ms.meanshift_cpp(coords, g.astype("float32"), 3, 30, 30, 10.0, 0.3,
                       0, np.array([]), 1.0, 0,
                       float(x.min()), float(x.max()),
                       float(y.min()), float(y.max()))
m = np.reshape(np.asarray(out), (900, -1))
u = np.unique(np.round(m[:,:2],0), axis=0)
print(f"  modes.shape = {m.shape}")
print(f"  x  {m[:,0].min():.3f} ~ {m[:,0].max():.3f}")
print(f"  unique modes = {len(u)}")
if len(u) >= 3:
    print(f"  ✓✓✓ compile succeeded! mode coords = {u[:5].tolist()}")
    print("  -> now you can run boms_xenium_2_run.py")
else:
    print(f"  ✘ still collapsed into {len(u)} points.")
    print("    compile passed but result is wrong; possibly a bug in the source or a compile-option issue.")
    print("    paste the compile log (output of [3]) back for further investigation.")
PYEOF

echo
echo "Done."
