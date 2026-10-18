#!/usr/bin/env bash
# =============================================================================
# BOMS C++ 扩展从头重编译
# =============================================================================
# 背景：预编译的 boms_wrapper.*.so 在本机段错误（合成数据都塌成单点 +
#       Segmentation fault），二进制不兼容。老服务器已不可用，只能本地重编。
#
# 这个脚本做四件事：
#   1. 探明源码位置和编译系统（Cython? pybind11? cppimport?）
#   2. 检查编译器 / OpenMP / numpy 头文件是否齐
#   3. 从头编译
#   4. 用合成数据验证（3 个点团必须给 ≥3 个 mode，不能塌成 1 个）
#
# 用法：
#   conda activate bboms
#   bash rebuild_boms.sh            # 全自动
#   bash rebuild_boms.sh --probe    # 只探测，不编译
# =============================================================================
set -o pipefail

PROBE_ONLY=0
[ "$1" = "--probe" ] && PROBE_ONLY=1

SEP="========================================================================"
PKG_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/boms"
SRC_GUESS_1="/data/qiuyijia/boms"
SRC_GUESS_2="/data/qiuyijia/boms/code"

echo "$SEP"
echo "① 探测源码与编译系统"
echo "$SEP"
echo "  CONDA_PREFIX = $CONDA_PREFIX"
echo "  已装 boms 包 = $PKG_DIR"
echo

# 已安装包里的 C++ 源
echo "  已装包内的 C++/pyx 源文件："
find "$PKG_DIR" -maxdepth 2 \( -name "*.cpp" -o -name "*.pyx" -o -name "*.h" \
     -o -name "*.hpp" -o -name "*.cc" \) 2>/dev/null | while read f; do
    echo "    $f  [$(du -h "$f" | cut -f1)]"
done

# 找源码仓库（setup.py / pyproject.toml）
echo
echo "  搜索源码仓库（含 setup.py 或 meanshift 源码）："
for d in "$SRC_GUESS_1" "$SRC_GUESS_2" "$HOME/boms" ; do
    [ -d "$d" ] || continue
    for marker in setup.py pyproject.toml meson.build; do
        if [ -f "$d/$marker" ]; then
            echo "    ✓ $d/$marker"
        fi
    done
    # 找 meanshift 的 C++ 源
    find "$d" -maxdepth 3 \( -name "*meanshift*" -o -name "*wrapper*" \) \
         \( -name "*.cpp" -o -name "*.pyx" -o -name "*.h" -o -name "*.hpp" \) \
         2>/dev/null | head -10 | while read f; do
        echo "      源: $f"
    done
done

echo
echo "$SEP"
echo "② 检查编译工具链"
echo "$SEP"
echo -n "  gcc      : "; gcc --version 2>/dev/null | head -1 || echo "✘ 缺"
echo -n "  g++      : "; g++ --version 2>/dev/null | head -1 || echo "✘ 缺"
echo -n "  cython   : "; python -c "import Cython; print(Cython.__version__)" 2>/dev/null || echo "✘ 缺 (pip install cython)"
echo -n "  pybind11 : "; python -c "import pybind11; print(pybind11.__version__)" 2>/dev/null || echo "（未装，可能不需要）"
echo -n "  numpy hdr: "; python -c "import numpy; print(numpy.get_include())" 2>/dev/null

# OpenMP 头 + 库
echo -n "  omp.h    : "
echo '#include <omp.h>
int main(){return omp_get_num_threads();}' > /tmp/_omptest.c
if gcc -fopenmp /tmp/_omptest.c -o /tmp/_omptest 2>/tmp/_omperr; then
    echo "✓ 可编译 OpenMP"
else
    echo "✘ OpenMP 编译失败："; cat /tmp/_omperr
fi

echo -n "  libgomp  : "; ls "$CONDA_PREFIX"/lib/libgomp.so* 2>/dev/null | head -1 || echo "系统路径"

if [ "$PROBE_ONLY" = "1" ]; then
    echo
    echo "  --probe 模式，到此为止。把上面的输出贴回来，据此定编译命令。"
    exit 0
fi

echo
echo "$SEP"
echo "③ 编译"
echo "$SEP"

# 优先：如果找到源码仓库，用它的 setup.py 重装
BUILT=0
for d in "$SRC_GUESS_1" "$SRC_GUESS_2" ; do
    if [ -f "$d/setup.py" ] || [ -f "$d/pyproject.toml" ]; then
        echo "  → 从源码仓库重装: $d"
        echo "    先卸载旧的预编译包..."
        pip uninstall -y boms 2>/dev/null
        cd "$d"
        # 强制重编译，不用缓存的 wheel
        echo "    pip install -e . --no-build-isolation --force-reinstall ..."
        if CC=gcc CXX=g++ CFLAGS="-fopenmp -O3" LDFLAGS="-fopenmp" \
           pip install -e . --no-build-isolation --force-reinstall 2>&1 | tail -40; then
            BUILT=1
            break
        fi
    fi
done

# 兜底：从 GitHub 拉源码
if [ "$BUILT" = "0" ]; then
    echo
    echo "  本地没找到可编译的源码仓库，从 GitHub 拉取..."
    CLONE_DIR="/data/qiuyijia/boms/boms_src_$(date +%s)"
    if git clone https://github.com/sciai-lab/boms "$CLONE_DIR" 2>&1 | tail -5; then
        cd "$CLONE_DIR"
        pip uninstall -y boms 2>/dev/null
        echo "    从 $CLONE_DIR 编译安装..."
        CC=gcc CXX=g++ CFLAGS="-fopenmp -O3" LDFLAGS="-fopenmp" \
           pip install -e . --no-build-isolation 2>&1 | tail -40
        BUILT=1
    else
        echo "  ✘ git clone 失败 —— 检查网络能否访问 github.com"
        echo "    （如果网络受限，需要联系管理员放行 github.com）"
        exit 1
    fi
fi

echo
echo "$SEP"
echo "④ 验证：3 个点团必须给 ≥3 个 mode"
echo "$SEP"
python - <<'PYEOF'
import numpy as np
try:
    import boms
    import boms_wrapper as ms
except Exception as e:
    print(f"  ✘ import 失败: {e}")
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
print(f"  唯一 mode 数 = {len(u)}")
if len(u) >= 3:
    print(f"  ✓✓✓ 编译成功！mode 坐标 = {u[:5].tolist()}")
    print("  → 现在可以跑 boms_xenium_2_run.py 了")
else:
    print(f"  ✘ 仍然塌成 {len(u)} 个点。")
    print("    编译过了但结果不对，可能是源码本身的 bug 或编译选项问题。")
    print("    把编译日志（③ 的输出）贴回来进一步查。")
PYEOF

echo
echo "完成。"