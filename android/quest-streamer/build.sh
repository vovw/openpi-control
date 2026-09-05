#!/usr/bin/env bash
# Standalone NativeActivity build: no Gradle, Java sources, or Unity install.
set -euo pipefail
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME}"
: "${ANDROID_SDK_ROOT:?set ANDROID_SDK_ROOT}"
: "${JAVA_HOME:?set JAVA_HOME}"
export PATH="$JAVA_HOME/bin:$PATH"
: "${OPENXR_AAR_DIR:?set OPENXR_AAR_DIR to the extracted Khronos loader AAR}"
quest_project=$(cd "$(dirname "$0")" && pwd)
quest_build=${QUEST_BUILD_DIR:-"$quest_project/build"}
quest_tools="$ANDROID_SDK_ROOT/build-tools/35.0.0"
mkdir -p "$quest_build/apk/lib/arm64-v8a"
cmake -S "$quest_project/app/src/main/cpp" -B "$quest_build/native" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENXR_INCLUDE_DIR="$OPENXR_AAR_DIR/prefab/modules/headers/include" \
    -DOPENXR_LOADER_LIBRARY="$OPENXR_AAR_DIR/jni/arm64-v8a/libopenxr_loader.so"
cmake --build "$quest_build/native" --parallel 4
cp "$quest_build/native/libopenpi_quest.so" "$quest_build/apk/lib/arm64-v8a/"
cp "$OPENXR_AAR_DIR/jni/arm64-v8a/libopenxr_loader.so" "$quest_build/apk/lib/arm64-v8a/"
"$quest_tools/aapt2" link -o "$quest_build/base.apk" \
    -I "$ANDROID_SDK_ROOT/platforms/android-32/android.jar" \
    --manifest "$quest_project/app/src/main/AndroidManifest.xml"
cp "$quest_build/base.apk" "$quest_build/unsigned.apk"
(cd "$quest_build/apk" && zip -q -u "$quest_build/unsigned.apk" lib/arm64-v8a/*.so)
"$quest_tools/zipalign" -f -p 4 "$quest_build/unsigned.apk" "$quest_build/aligned.apk"
if [[ ! -f "$quest_build/debug.keystore" ]]; then
    "$JAVA_HOME/bin/keytool" -genkeypair -keystore "$quest_build/debug.keystore" \
        -storepass android -keypass android -alias androiddebugkey -keyalg RSA -keysize 2048 \
        -validity 3650 -dname 'CN=OpenPI Local Development'
fi
"$quest_tools/apksigner" sign --ks "$quest_build/debug.keystore" \
    --ks-pass pass:android --ks-key-alias androiddebugkey \
    --out "$quest_build/openpi-quest-streamer.apk" "$quest_build/aligned.apk"
"$quest_tools/apksigner" verify "$quest_build/openpi-quest-streamer.apk"
echo "APK: $quest_build/openpi-quest-streamer.apk"
