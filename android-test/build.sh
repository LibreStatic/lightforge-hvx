#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Hand-built APK (no Gradle): aapt2 + javac + d8 + zipalign + apksigner.
# Output: android-test/out/lfhvx-test.apk
set -euo pipefail
cd "$(dirname "$0")"
SDK=${ANDROID_SDK:-$HOME/Android/Sdk}
BT=${BUILD_TOOLS:-$SDK/build-tools/36.0.0}
JAR=${ANDROID_JAR:-$SDK/platforms/android-36/android.jar}
NDK=${NDK:-$SDK/ndk/28.2.13676358}
CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang
KS=${KEYSTORE:-$HOME/.android/debug.keystore}
O=out

rm -rf "$O" && mkdir -p "$O/lib/arm64-v8a" "$O/classes" "$O/dex"

# Hexagon skel (upstream clang/lld) via the top-level Makefile.
make -C .. build/liblfhvx_skel.so >/dev/null
cp ../build/liblfhvx_skel.so "$O/lib/arm64-v8a/"

# arm64 JNI library.
"$CC" -O2 -Wall -Wextra -fPIC -shared -Wl,-z,max-page-size=16384 \
    -o "$O/lib/arm64-v8a/liblfhvx_jni.so" jni/lfhvx_jni.c -llog -ldl

# Manifest-only resource table.
"$BT/aapt2" link --manifest AndroidManifest.xml -I "$JAR" -o "$O/base.apk"

javac -nowarn -Xlint:-options -source 17 -target 17 -classpath "$JAR" -d "$O/classes" \
    $(find src -name '*.java')
"$BT/d8" --lib "$JAR" --min-api 31 --output "$O/dex" $(find "$O/classes" -name '*.class')

cp "$O/base.apk" "$O/unsigned.apk"
(cd "$O/dex" && zip -q ../unsigned.apk classes.dex)
(cd "$O" && zip -q unsigned.apk lib/arm64-v8a/*.so)
"$BT/zipalign" -f -P 16 4 "$O/unsigned.apk" "$O/aligned.apk"
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android \
    --out "$O/lfhvx-test.apk" "$O/aligned.apk"
"$BT/apksigner" verify "$O/lfhvx-test.apk"
unzip -l "$O/lfhvx-test.apk"
