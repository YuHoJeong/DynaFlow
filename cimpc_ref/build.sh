#!/usr/bin/env bash
# CI-MPC 원본 소스를 건드리지 않고 빌드한다.
# Eigen 3.4 는 rank-3 tensor slice 를 rank-2 에 바로 대입하지 못해서
# contactSolverGradient.cpp 사본에 reshape 만 넣어 쓴다 (원본은 구버전 Eigen 전제).
set -e
SRC="${CIMPC_SRC:-$HOME/workspace/general_contact_TO/contact_solving_gradient}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/build"
python - "$SRC/lib/contactSolverGradient.cpp" "$HERE/build/contactSolverGradient_patched.cpp" << 'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
hdr = '#include "contactSolverGradient.h"'
s = s.replace(hdr, hdr + '\n\n// [빌드 패치] Eigen 3.4: rank-3 slice -> rank-2 대입 불가. reshape 로 마지막 차원(1) 제거.\nstatic const std::array<long, 2> extent2d = {6, nv + 4};\n', 1)
n = s.count('.slice(offset, extent);')
s = s.replace('.slice(offset, extent);', '.slice(offset, extent).reshape(extent2d);')
open(dst, 'w').write(s)
print(f"  reshape 패치 {n}곳 적용")
PY
cmake -S "$HERE" -B "$HERE/build" -DCIMPC_SRC="$SRC" > "$HERE/build/cmake.log" 2>&1
cmake --build "$HERE/build" -j8 > "$HERE/build/make.log" 2>&1
echo "빌드 완료: $HERE/build/dump_gradient"
