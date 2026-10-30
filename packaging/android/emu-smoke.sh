#!/bin/bash
# PXA Control for Android: headless emulator smoke (needs /dev/kvm). Installs the APK, starts it (rig picker), then
# opens a PXA Control started in this container with --lan (the emulator reaches the container at 10.0.2.2) and checks
# the signed-in page reached the WebView (Control's request log + screenshots).
#   docker run --rm --device /dev/kvm -v <repo>:/src:ro -v <sdk>:/sdk -v <out>:/out eclipse-temurin:17-jdk-jammy \
#       bash /src/packaging/android/emu-smoke.sh /out/<apk>
set -uo pipefail
APK=${1:?apk}; SDK=/sdk; OUT=/out/emu; mkdir -p $OUT
export ANDROID_SDK_ROOT=$SDK ANDROID_HOME=$SDK ANDROID_AVD_HOME=/tmp/avd DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq python3 >/dev/null 2>&1; apt-get install -y -qq libpulse0 libgl1 libnss3 libxcomposite1 libxcursor1 libxi6 libxtst6 libxdamage1 libxrandr2 libasound2 libdbus-1-3 >/dev/null 2>&1
SM="$SDK/cmdline-tools/latest/bin/sdkmanager --sdk_root=$SDK"
API=${PXA_EMU_API:-34}; IMG="system-images;android-$API;default;x86_64"   # API 30s stock WebView (Chrome 83) lacks replaceChildren: Control needs WebView 86+
[ -d "$SDK/emulator" ] && [ -d "$SDK/system-images/android-$API/default/x86_64" ] || { yes | $SM --licenses >/dev/null 2>&1; $SM emulator platform-tools "$IMG" > $OUT/sdkmanager.log 2>&1; }
[ -x "$SDK/platform-tools/adb" ] || $SM platform-tools >> $OUT/sdkmanager.log 2>&1
ADB=$SDK/platform-tools/adb
echo no | $SDK/cmdline-tools/latest/bin/avdmanager create avd -n pxa -k "$IMG" -d pixel --force > $OUT/avd.log 2>&1
$SDK/emulator/emulator -avd pxa -no-window -no-audio -no-boot-anim -gpu swiftshader_indirect -no-snapshot -accel on > $OUT/emulator.log 2>&1 &
$ADB wait-for-device
for i in $(seq 1 120); do [ "$($ADB shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = 1 ] && break; sleep 3; done
echo "boot_completed after ~$((i*3)) s"
# a real Control, --lan, fixed token, in this container
mkdir -p /tmp/home/cfg && export HOME=/tmp/home PXA_CONTROL_CONFIG_DIR=/tmp/home/cfg PXA_LAUNCH_FAKE_GPUS=1x600 PXA_CONTROL_TELEMETRY=0 PXA_CONTROL_DISCOVER=0 PXA_CONTROL_TOKEN=emuSmokeToken0123456789
python3 /src/tools/pxa_control.py --lan --no-browser --no-tray --port 7877 > $OUT/control.log 2>&1 &
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:7877/api/health && break; sleep 1; done
$ADB install -r "$APK" | tee $OUT/install.txt
$ADB shell am start -W -n net.pxanetwork.control/.MainActivity | tee $OUT/start.txt
sleep 8; $ADB exec-out screencap -p > $OUT/1-picker.png
# open the rig the way a user would (the picker builds the same URL): the activity's WebView follows it
$ADB shell am start -W -a android.intent.action.VIEW -d "http://10.0.2.2:7877/?token=emuSmokeToken0123456789" -n net.pxanetwork.control/.MainActivity >/dev/null 2>&1
sleep 2
# type into the picker instead (the reliable path for a launcher-only activity): tap address field, type, open
$ADB shell input keyevent KEYCODE_BACK >/dev/null 2>&1
sleep 1; $ADB shell am start -W -n net.pxanetwork.control/.MainActivity >/dev/null
sleep 3; $ADB exec-out uiautomator dump /dev/tty > $OUT/ui1.xml 2>/dev/null
python3 - "$OUT/ui1.xml" > $OUT/addr-xy.txt <<'PY'
import re, sys
x = open(sys.argv[1], errors="replace").read()
for m in re.finditer(r'<node [^>]*?(?:resource-id="addr"|hint="192\.168[^"]*"|text="192\.168[^"]*")[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', x):
    a, b, c, d = map(int, m.groups()); print((a + c) // 2, (b + d) // 2); break
PY
XY=$(cat $OUT/addr-xy.txt); [ -n "$XY" ] || XY="540 1055"      # the address field on a 1080x1920 pixel AVD
$ADB shell input tap $XY; sleep 1
$ADB shell input text "10.0.2.2:7877"; sleep 1
$ADB shell input keyevent KEYCODE_TAB; sleep 1; $ADB shell input text "emuSmokeToken0123456789"; sleep 1
$ADB shell input keyevent KEYCODE_ENTER                          # submits the form (Open)
sleep 15; $ADB exec-out screencap -p > $OUT/2-control.png
$ADB exec-out uiautomator dump /dev/tty > $OUT/ui2.xml 2>/dev/null
# Control's sign-in page (token not carried): type the token into its field, as a user would
if grep -q "enter the token" $OUT/ui2.xml; then
  XY2=$(python3 -c "
import re,sys
x=open('$OUT/ui2.xml',errors='replace').read()
m=re.search(r'class=\"android.widget.EditText\"[^>]*bounds=\"\[(\d+),(\d+)\]\[(\d+),(\d+)\]\"',x)
print('' if not m else '%d %d'%((int(m.group(1))+int(m.group(3)))//2,(int(m.group(2))+int(m.group(4)))//2))")
  echo "login field: $XY2"
  [ -n "$XY2" ] && { $ADB shell input tap $XY2; sleep 1; $ADB shell input text "emuSmokeToken0123456789"; sleep 1; $ADB shell input keyevent KEYCODE_ENTER; sleep 20; }
  $ADB exec-out screencap -p > $OUT/3-signed-in.png
  $ADB exec-out uiautomator dump /dev/tty > $OUT/ui3.xml 2>/dev/null
  grep -o 'text="[^"]\{3,\}"' $OUT/ui3.xml | head -30 > $OUT/signed-in.txt
fi
grep -o 'text="[^"]\{3,\}"' $OUT/ui2.xml | head -30 > $OUT/requests.txt
curl -s -o /dev/null -w "control health from the container: %{http_code}\n" -H "X-PXA-Token: emuSmokeToken0123456789" http://127.0.0.1:7877/api/health >> $OUT/requests.txt
$ADB logcat -d | grep -i -E "chromium|pxanetwork" | tail -40 > $OUT/logcat.txt
$ADB emu kill >/dev/null 2>&1
echo "picker xy: $XY"; cat $OUT/requests.txt; echo "== after sign-in"; cat $OUT/signed-in.txt 2>/dev/null
