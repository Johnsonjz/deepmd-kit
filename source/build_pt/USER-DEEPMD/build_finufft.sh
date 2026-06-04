#!/bin/bash
set -euo pipefail

REPO_URL="https://github.com/flatironinstitute/finufft.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/.finufft-src"
BUILD_DIR="${SRC_DIR}/build"
INSTALL_PREFIX=""
CMAKE_BUILD_TYPE="Release"
UPDATE_REPO=0
CLEAN_BUILD=0

if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
else
  JOBS=4
fi

usage() {
  cat <<'EOF'
Build FINUFFT shared library for SOG backend.

Usage:
  ./build_finufft.sh [options]

Options:
  --repo-url URL         FINUFFT git URL.
  --src-dir DIR          Source directory for finufft clone.
  --build-dir DIR        Build directory.
  --install-prefix DIR   Optional install prefix (cmake --install).
  --build-type TYPE      CMAKE_BUILD_TYPE (default: Release).
  --jobs N               Build parallel jobs.
  --update               Run git pull --ff-only if source exists.
  --clean                Remove build directory before configure.
  -h, --help             Show this help message.

Output:
  Prints absolute path to libfinufft.so when build succeeds.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --src-dir)
      SRC_DIR="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --install-prefix)
      INSTALL_PREFIX="$2"
      shift 2
      ;;
    --build-type)
      CMAKE_BUILD_TYPE="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --update)
      UPDATE_REPO=1
      shift
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  echo "Cloning FINUFFT from ${REPO_URL} ..."
  rm -rf "${SRC_DIR}"
  git clone "${REPO_URL}" "${SRC_DIR}"
fi

if [[ ${UPDATE_REPO} -eq 1 ]]; then
  echo "Updating FINUFFT source in ${SRC_DIR} ..."
  git -C "${SRC_DIR}" pull --ff-only
fi

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

echo "Configuring FINUFFT ..."
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DBUILD_SHARED_LIBS=ON \
  -DFINUFFT_STATIC_LINKING=OFF \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

echo "Building FINUFFT ..."
cmake --build "${BUILD_DIR}" -j "${JOBS}"

if [[ -n "${INSTALL_PREFIX}" ]]; then
  echo "Installing FINUFFT to ${INSTALL_PREFIX} ..."
  cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}"
fi

LIB_PATH=""
for candidate in \
  "${BUILD_DIR}/src/libfinufft.so" \
  "${BUILD_DIR}/lib/libfinufft.so" \
  "${BUILD_DIR}/lib64/libfinufft.so"; do
  if [[ -f "${candidate}" ]]; then
    LIB_PATH="$(cd "$(dirname "${candidate}")" && pwd)/$(basename "${candidate}")"
    break
  fi
done

if [[ -z "${LIB_PATH}" ]]; then
  echo "Build finished but libfinufft.so not found. Check ${BUILD_DIR}." >&2
  exit 3
fi

echo ""
echo "FINUFFT build done:"
echo "  ${LIB_PATH}"
echo ""
echo "Use one of the following for SOG:"
echo "  export DP_SOG_FINUFFT_LIBRARY=${LIB_PATH}"
echo "  kspace_style sog ... finufft_library ${LIB_PATH}"
