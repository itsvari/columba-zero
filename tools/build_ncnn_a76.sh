#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
profile=mac-a76compat
prefix=
source_directory=
parallel_jobs=4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) profile="$2"; shift 2 ;;
        --prefix) prefix="$2"; shift 2 ;;
        --source) source_directory="$2"; shift 2 ;;
        --jobs) parallel_jobs="$2"; shift 2 ;;
        --help)
            echo 'Build ncnn tag 20260526 plus the ctypes benchmark bridge. Options: --profile mac-a76compat|xcng|pi5 --prefix DIR --source DIR --jobs N (default 4).'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$profile" in mac-a76compat|xcng|pi5) ;; *) echo 'Invalid profile' >&2; exit 2 ;; esac
prefix="${prefix:-$repository_root/native/build/ncnn-$profile}"
mkdir -p "$prefix"
prefix="$(cd "$prefix" && pwd)"
source_directory="${source_directory:-$prefix/source}"
if [[ ! -d "$source_directory/.git" ]]; then
    git clone --depth 1 --branch 20260526 https://github.com/Tencent/ncnn.git "$source_directory"
fi
expected_commit=e54f7b1f88434e1d844ea0551b880a1cfb079ce1
[[ "$(git -C "$source_directory" rev-parse HEAD)" == "$expected_commit" ]] || {
    echo 'Source does not match pinned ncnn tag 20260526' >&2; exit 1;
}
flags=(-DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF
    -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_TOOLS=ON -DNCNN_PYTHON=OFF
    -DNCNN_SHARED_LIB=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DNCNN_ARM82=ON -DNCNN_ARM82DOT=ON)
# disable every post-A76 ARM option actually declared by this release.
while IFS= read -r option; do flags+=("-D$option=OFF"); done < <(
    sed -nE 's/.*option\((NCNN_ARM(82FP16FML|84[^ ]*|86[^ ]*|[0-9]+SME[^ ]*)) .*/\1/p' "$source_directory/CMakeLists.txt"
)
if [[ "$profile" == mac-a76compat ]]; then
    openmp_prefix="$(brew --prefix libomp 2>/dev/null || true)"
    [[ -n "$openmp_prefix" ]] || { echo 'mac profile requires Homebrew libomp' >&2; exit 1; }
    flags+=(-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64 -DOpenMP_C_FLAGS=-Xpreprocessor\ -fopenmp
        -DOpenMP_CXX_FLAGS=-Xpreprocessor\ -fopenmp -DOpenMP_C_LIB_NAMES=omp
        -DOpenMP_CXX_LIB_NAMES=omp "-DOpenMP_omp_LIBRARY=$openmp_prefix/lib/libomp.dylib"
        "-DCMAKE_C_FLAGS=-I$openmp_prefix/include" "-DCMAKE_CXX_FLAGS=-I$openmp_prefix/include")
fi
cmake -S "$source_directory" -B "$prefix/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" "${flags[@]}"
cmake --build "$prefix/build" --parallel "$parallel_jobs"
cmake --install "$prefix/build"
cmake -S "$repository_root/tools/bench_ncnn" -B "$prefix/bridge-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_PREFIX_PATH="$prefix" "${flags[@]}"
cmake --build "$prefix/bridge-build" --parallel "$parallel_jobs"
cmake --install "$prefix/bridge-build"
{
    echo "profile=$profile"
    echo "tag=20260526 commit=$expected_commit pip=1.0.20260526"
    sed -nE '/^NCNN_(ARM|OPENMP|VULKAN).*:(BOOL|UNINITIALIZED)=/p' "$prefix/build/CMakeCache.txt"
    sed -nE '/^#define NCNN_ARM/p' "$prefix/build/src/platform.h"
    echo 'Compiled ARM kernel objects:'
    ar -t "$prefix/lib/libncnn.a" | sed -nE '/arm.*(asimd|fp16|bf16|i8mm|sve|sme)/p'
    echo 'Compiled kernel symbols:'
    nm -g "$prefix/lib/libncnn.a" 2>/dev/null | sed -nE '/asimdhp|asimddp|fp16fml|bf16|i8mm|sve|sme/p' | sed -n '1,100p' || true
} > "$prefix/build-evidence.txt"
[[ -f "$prefix/lib/cmake/ncnn/ncnnConfig.cmake" ]]
awk '
    $1 == "#define" && $2 ~ /^NCNN_ARM/ {
        expected = ($2 == "NCNN_ARM82" || $2 == "NCNN_ARM82DOT") ? 1 : 0
        if ($3 != expected) { print "Unexpected ARM feature: " $2 "=" $3; failed = 1 }
    }
    END { exit failed }
' "$prefix/build/src/platform.h"
echo "Installed $prefix; evidence: $prefix/build-evidence.txt"
