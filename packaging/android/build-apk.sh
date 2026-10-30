#!/bin/bash
# PXA Control for Android: build the WebView wrapper APK with the Android command-line tools only (no Android Studio,
# no Gradle), inside a JDK container. Reproducible: pinned tool versions, a fixed source list, a throwaway debug key.
#   docker run --rm -v <repo>:/src:ro -v <sdk cache>:/sdk -v <out>:/out eclipse-temurin:17-jdk-jammy \
#       bash /src/packaging/android/build-apk.sh
# Output: /out/pxa-control-<ver>-android.apk (+ .sha256) and /out/debug.keystore (throwaway, made on first build).
set -euo pipefail
SRC=/src; A=$SRC/packaging/android; SDK=/sdk; OUT=/out; W=/tmp/apk
CLT_ZIP=commandlinetools-linux-11076708_latest.zip
PLATFORM="platforms;android-34"; BT_VER=34.0.0
VER="${PXA_APK_VERSION:-4.0-pre}"
export DEBIAN_FRONTEND=noninteractive
command -v unzip >/dev/null && command -v zip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip zip >/dev/null; }
mkdir -p "$SDK" "$OUT"
if [ ! -x "$SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
  mkdir -p "$SDK/dl" && cd "$SDK/dl"
  [ -f "$CLT_ZIP" ] || curl -sSfLO "https://dl.google.com/android/repository/$CLT_ZIP"
  sha256sum "$CLT_ZIP" | tee "$SDK/dl/$CLT_ZIP.sha256"
  rm -rf "$SDK/cmdline-tools" && mkdir -p "$SDK/cmdline-tools" && unzip -q "$CLT_ZIP" -d "$SDK/cmdline-tools" \
    && mv "$SDK/cmdline-tools/cmdline-tools" "$SDK/cmdline-tools/latest"
fi
SM="$SDK/cmdline-tools/latest/bin/sdkmanager --sdk_root=$SDK"
if [ ! -f "$SDK/platforms/android-34/android.jar" ] || [ ! -x "$SDK/build-tools/$BT_VER/aapt2" ]; then
  yes | $SM --licenses >/dev/null 2>&1 || true
  $SM "$PLATFORM" "build-tools;$BT_VER" > "$SDK/sdkmanager.log" 2>&1
fi
BT=$SDK/build-tools/$BT_VER; JAR=$SDK/platforms/android-34/android.jar
rm -rf "$W" && mkdir -p "$W"/{res/mipmap-xxxhdpi,res/values,assets,gen,classes,dex,compiled}
cp "$A/res/values/strings.xml" "$W/res/values/"
cp "$SRC/tools/pxa_control_ui/icon-192.png" "$W/res/mipmap-xxxhdpi/ic_launcher.png"
cp "$SRC/tools/pxa_control_ui/hosts.html" "$W/assets/hosts.html"
sed "s/android:versionName=\"[^\"]*\"/android:versionName=\"$VER\"/" "$A/AndroidManifest.xml" > "$W/AndroidManifest.xml"
"$BT/aapt2" compile --dir "$W/res" -o "$W/compiled/res.zip"
"$BT/aapt2" link -I "$JAR" --manifest "$W/AndroidManifest.xml" -A "$W/assets" --java "$W/gen" \
    -o "$W/base.apk" "$W/compiled/res.zip" --min-sdk-version 24 --target-sdk-version 34
find "$A/src" "$W/gen" -name '*.java' | sort > "$W/sources.txt"
javac -encoding UTF-8 --release 8 -classpath "$JAR" -d "$W/classes" @"$W/sources.txt" 2>&1 | grep -v "warning: \[options\]" || true
"$BT/d8" --release --min-api 24 --lib "$JAR" --output "$W/dex" $(find "$W/classes" -name '*.class' | sort)
cp "$W/base.apk" "$W/unsigned.apk" && (cd "$W/dex" && TZ=UTC touch -t 198001010000 classes.dex && TZ=UTC zip -q -X "$W/unsigned.apk" classes.dex)
"$BT/zipalign" -f -p 4 "$W/unsigned.apk" "$W/aligned.apk"
KS=$OUT/debug.keystore
[ -f "$KS" ] || keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias pxadebug -keyalg RSA \
    -keysize 2048 -validity 3650 -dname "CN=PXA Control debug (throwaway), O=PXA Network" >/dev/null 2>&1
NAME=pxa-control-$VER-android.apk
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --ks-key-alias pxadebug \
    --out "$OUT/$NAME" "$W/aligned.apk"
# full output to a file first: `| head` closes the pipe early and pipefail would fail the build on SIGPIPE
"$BT/apksigner" verify --print-certs "$OUT/$NAME" > "$W/verify.txt" && head -3 "$W/verify.txt"
"$BT/aapt2" dump badging "$OUT/$NAME" > "$W/badging.txt" && grep -E "^(package|sdkVersion|targetSdkVersion|uses-permission):" "$W/badging.txt"
(cd "$OUT" && sha256sum "$NAME" > "$NAME.sha256" && ls -la "$NAME")
