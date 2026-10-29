#!/usr/bin/env bash
#
# Replace the prebuilt libraries in xgboost-sys/lib with the official XGBoost builds from PyPI.
#
# The wheels hold the plain C library; Python is not involved. They are patched to behave like a
# standalone library: Linux gets an unversioned soname and the system libgomp instead of the
# wheel's bundled copy, macOS an unversioned install name, and Windows a generated import library.
#
# Usage:
#   scripts/fetch-prebuilt-libs.sh [version]     (defaults to the xgboost-sys/xgboost submodule tag)
#
# Requires macOS (install_name_tool, codesign), Homebrew llvm (llvm-readobj, llvm-lib), uvx
# (for patchelf), curl, unzip and python3. Afterwards, update the checksums in build.rs with the
# printed values and bump LIB_TAG.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIB_DIR="$REPO_ROOT/xgboost-sys/lib"
VERSION=${1:-$(git -C "$REPO_ROOT/xgboost-sys/xgboost" describe --tags --exact-match | sed 's/^v//')}
LLVM_BIN=${LLVM_BIN:-$(brew --prefix llvm)/bin}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Downloads the wheel for one PyPI package/platform tag, verifies it against PyPI's digest and
# unpacks it into $WORK/<platform>.
fetch_wheel() {
    local package=$1 wheel_tag=$2 platform=$3
    local dir="$WORK/$platform"
    mkdir -p "$dir"
    read -r url sha < <(curl -sf "https://pypi.org/pypi/$package/$VERSION/json" | python3 -I -c "
import json, sys
urls = [u for u in json.load(sys.stdin)['urls'] if '$wheel_tag' in u['filename']]
if not urls: sys.exit('no $wheel_tag wheel for $package $VERSION')
print(urls[0]['url'], urls[0]['digests']['sha256'])")
    curl -sfL "$url" -o "$dir/wheel.whl"
    echo "$sha  $dir/wheel.whl" | shasum -a 256 -c --quiet
    unzip -qo "$dir/wheel.whl" -d "$dir"
}

patch_linux() {
    local platform=$1 so="$WORK/$1/xgboost/lib/libxgboost.so"
    local gomp
    gomp=$("$LLVM_BIN/llvm-readobj" --needed-libs "$so" | grep -oE 'libgomp[^ ]*\.so[.0-9]*')
    uvx patchelf --set-soname libxgboost.so --replace-needed "$gomp" libgomp.so.1 --remove-rpath "$so"
    cp "$so" "$LIB_DIR/$platform/libxgboost.so"
}

patch_mac() {
    local dylib="$WORK/mac_arm64/xgboost/lib/libxgboost.dylib"
    install_name_tool -id @rpath/libxgboost.dylib "$dylib"
    codesign --force --sign - "$dylib"
    cp "$dylib" "$LIB_DIR/mac_arm64/libxgboost.dylib"
}

patch_windows() {
    local dll="$WORK/win_amd64/xgboost/lib/xgboost.dll" def="$WORK/win_amd64/xgboost.def"
    {
        echo "LIBRARY xgboost.dll"
        echo "EXPORTS"
        "$LLVM_BIN/llvm-readobj" --coff-exports "$dll" | awk '/^ *Name: /{print "    " $2}'
    } > "$def"
    "$LLVM_BIN/llvm-lib" /nologo /machine:x64 /def:"$def" /out:"$LIB_DIR/win_amd64/xgboost.lib"
    cp "$dll" "$LIB_DIR/win_amd64/xgboost.dll"
}

echo "Fetching XGBoost $VERSION from PyPI"
fetch_wheel xgboost-cpu manylinux_2_28_x86_64 linux_amd64
fetch_wheel xgboost-cpu manylinux_2_28_aarch64 linux_arm64
fetch_wheel xgboost-cpu win_amd64 win_amd64
fetch_wheel xgboost macosx_12_0_arm64 mac_arm64

patch_linux linux_amd64
patch_linux linux_arm64
patch_mac
patch_windows

echo "SHA-256 for build.rs:"
(cd "$LIB_DIR" && shasum -a 256 linux_amd64/libxgboost.so linux_arm64/libxgboost.so \
    mac_arm64/libxgboost.dylib win_amd64/xgboost.dll win_amd64/xgboost.lib)
