#!/usr/bin/env bash
set -euo pipefail

profile="${1:-host}"
build_ncnn=true
case "${2:-}" in
    "") ;;
    --opencv-only) build_ncnn=false ;;
    *) printf 'usage: build_dependencies.sh [PROFILE] [--opencv-only]\n' >&2; exit 2 ;;
esac
build_threads="${BUILD_THREADS:-4}"
native_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_directory="$native_directory/build"
ncnn_prefix="$build_directory/ncnn-$profile"
opencv_prefix="$build_directory/opencv-$profile"

mkdir -p "$build_directory/sources"
if [[ "$build_ncnn" == true && ! -d "$build_directory/sources/ncnn/.git" ]]; then
    git clone --depth 1 --branch 20260526 https://github.com/Tencent/ncnn.git \
        "$build_directory/sources/ncnn"
fi
if [[ ! -d "$build_directory/sources/opencv/.git" ]]; then
    git clone --depth 1 --branch 5.0.0 https://github.com/opencv/opencv.git \
        "$build_directory/sources/opencv"
fi

openmp_options=()
architecture_options=()
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
    architecture_options=(-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64)
fi
if [[ "$(uname -s)" == Darwin && -d /opt/homebrew/opt/libomp ]]; then
    openmp_options=(
        -DOpenMP_C_FLAGS=-Xpreprocessor\ -fopenmp
        -DOpenMP_CXX_FLAGS=-Xpreprocessor\ -fopenmp
        -DOpenMP_C_LIB_NAMES=omp
        -DOpenMP_CXX_LIB_NAMES=omp
        -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
        -DCMAKE_C_FLAGS=-I/opt/homebrew/opt/libomp/include
        -DCMAKE_CXX_FLAGS=-I/opt/homebrew/opt/libomp/include
    )
fi

if [[ "$build_ncnn" == true ]]; then
    cmake -S "$build_directory/sources/ncnn" -B "$build_directory/ncnn-$profile-build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ncnn_prefix" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DNCNN_VULKAN=OFF \
        -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_BUILD_TESTS=OFF "${openmp_options[@]}" "${architecture_options[@]}"
    cmake --build "$build_directory/ncnn-$profile-build" --parallel "$build_threads"
    cmake --install "$build_directory/ncnn-$profile-build"
fi

cmake -S "$build_directory/sources/opencv" -B "$build_directory/opencv-$profile-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$opencv_prefix" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIST=core,imgproc -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_LAPACK=OFF \
    -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_JPEG=OFF -DWITH_PNG=OFF \
    -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_AVIF=OFF -DWITH_OPENJPEG=OFF \
    -DWITH_OPENEXR=OFF -DWITH_JASPER=OFF -DWITH_PROTOBUF=OFF "${architecture_options[@]}"
cmake --build "$build_directory/opencv-$profile-build" --parallel "$build_threads"
cmake --install "$build_directory/opencv-$profile-build"

if [[ "$build_ncnn" == true ]]; then
    printf 'ncnn_DIR=%s/lib/cmake/ncnn\n' "$ncnn_prefix"
fi
printf 'OpenCV_DIR=%s/lib/cmake/opencv5\n' "$opencv_prefix"
