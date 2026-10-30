#!/bin/bash
# PXA Control for Android: headless emulator smoke of the offline/reconnect screen (needs /dev/kvm). Installs the APK,
# adds a rig whose Control is NOT running yet (10.0.2.2:7899 = this container), checks the app shows the picker's offline
# screen naming that rig (not a WebView error page), then starts a real Control there and checks the app reopens it by
# itself (Control's sign-in page shows up without a tap).
#   docker run --rm --device /dev/kvm -v <repo>:/src:ro -v <sdk>:/sdk -v <out>:/out eclipse-temurin:17-jdk-jammy \
#       bash /src/packaging/android/emu-offline-smoke.sh /out/<apk>
# Prints PASS/FAIL lines; exit 0 = both checks passed. Screens + ui dumps in /out/emu-offline.
set -uo pipefail
APK=${1:?apk}; SDK=/sdk; OUT=/out/emu-offline; mkdir -p $OUT
export ANDROID_SDK_ROOT=$SDK ANDROID_HOME=$SDK ANDROID_AVD_HOME=/tmp/avd DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq python3 curl >/dev/null 2>&1; apt-get install -y -qq libpulse0 libgl1 libnss3 libxcomposite1 libxcursor1 libxi6 libxtst6 libxdamage1 libxrandr2 libasound2 libdbus-1-3 >/dev/null 2>&1
SM="$SDK/cmdline-tools/latest/bin/sdkmanager --sdk_root=$SDK"
API=${PXA_EMU_API:-34}; IMG="system-images;android-$API;default;x86_64"
[ -d "$SDK/emulator" ] && [ -d "$SDK/system-images/android-$API/default/x86_64" ] || { yes | $SM --licenses >/dev/null 2>&1; $SM emulator platform-tools "$IMG" > $OUT/sdkmanager.log 2>&1; }
[ -x "$SDK/platform-tools/adb" ] || $SM platform-tools >> $OUT/sdkmanager.log 2>&1
ADB=$SDK/platform-tools/adb
echo no | $SDK/cmdline-tools/latest/bin/avdmanager create avd -n pxa -k "$IMG" -d pixel --force > $OUT/avd.log 2>&1
$SDK/emulator/emulator -avd pxa -no-window -no-audio -no-boot-anim -gpu swiftshader_indirect -no-snapshot -accel on > $OUT/emulator.log 2>&1 &
$ADB wait-for-device
for i in $(seq 1 120); do [ "$($ADB shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = 1 ] && break; sleep 3; done
echo "boot_completed after ~$((i*3)) s"
$ADB install --no-incremental -r "$APK" | tee $OUT/install.txt
$ADB shell am start -W -n net.pxanetwork.control/.MainActivity > $OUT/start.txt
dump() { $ADB exec-out uiautomator dump /dev/tty > $OUT/$1.xml 2>/dev/null; grep -o 'text="[^"]\{3,\}"' $OUT/$1.xml > $OUT/$1.txt; $ADB exec-out screencap -p > $OUT/$1.png; }
# a slow (loaded) emulator can raise "isn't responding" while the WebView starts cold: answer Wait, as a user would
waitfor() {  # waitfor <name> <regex> <seconds>
  local n=$1 re=$2 t=$3 k
  for k in $(seq 1 $((t / 5))); do
    dump $n
    if grep -q "isn't responding" $OUT/$n.txt; then
      W=$(python3 -c "
import re
x=open('$OUT/$n.xml',errors='replace').read()
m=re.search(r'text=\"Wait\"[^>]*bounds=\"\[(\d+),(\d+)\]\[(\d+),(\d+)\]\"',x)
print('' if not m else '%d %d'%((int(m.group(1))+int(m.group(3)))//2,(int(m.group(2))+int(m.group(4)))//2))")
      echo "ANR dialog in $n: Wait"; [ -n "$W" ] && $ADB shell input tap $W
      [ -f $OUT/anr-main.txt ] || { $ADB root >/dev/null 2>&1; sleep 2; $ADB shell 'cat $(ls -t /data/anr/* | head -1)' 2>/dev/null \
        | awk '/^"main"/,/^$/' | head -40 > $OUT/anr-main.txt; }
    elif grep -q -E "$re" $OUT/$n.txt; then return 0; fi
    sleep 5
  done
  return 1
}
if waitfor 1-picker "Add a rig" 120; then echo "picker up"; else
  echo "picker not seen in 120 s"; $ADB logcat -d > $OUT/logcat-full.txt; $ADB emu kill >/dev/null 2>&1; exit 3; fi
XY=$(python3 - "$OUT/1-picker.xml" <<'PY'
import re, sys
x = open(sys.argv[1], errors="replace").read()
for m in re.finditer(r'<node [^>]*?(?:resource-id="addr"|hint="192\.168[^"]*"|text="192\.168[^"]*")[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', x):
    a, b, c, d = map(int, m.groups()); print((a + c) // 2, (b + d) // 2); break
PY
); [ -n "$XY" ] || XY="540 1055"
$ADB shell input tap $XY; sleep 1; $ADB shell input text "10.0.2.2:7899"; sleep 1; $ADB shell input keyevent KEYCODE_ENTER
waitfor 2-offline "not answering" 90
rc=0
if grep -q "not answering" $OUT/2-offline.txt && grep -q "10.0.2.2:7899" $OUT/2-offline.txt; then echo "PASS offline screen names the rig"; else echo "FAIL offline screen (see 2-offline.txt)"; rc=1; fi
grep -q -i "webpage not available\|ERR_CONNECTION" $OUT/2-offline.txt && { echo "FAIL WebView error page shown"; rc=1; }
mkdir -p /tmp/home/cfg && export HOME=/tmp/home PXA_CONTROL_CONFIG_DIR=/tmp/home/cfg PXA_LAUNCH_FAKE_GPUS=1x600 PXA_CONTROL_TELEMETRY=0 PXA_CONTROL_DISCOVER=0 PXA_CONTROL_TOKEN=emuSmokeToken0123456789
python3 /src/tools/pxa_control.py --lan --no-browser --no-tray --port 7899 > $OUT/control.log 2>&1 &
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:7899/api/health && break; sleep 1; done
echo "control up after ~$i s; waiting for the app to reopen it by itself"
# Control's own page (its sign-in asks for the token) = the app reopened the rig without a tap
if waitfor 3-reopened "enter the token" 120 && ! grep -q "not answering" $OUT/3-reopened.txt; then
  echo "PASS reopened by itself"; head -8 $OUT/3-reopened.txt
else echo "FAIL not reopened in 120 s"; rc=1; fi
$ADB logcat -d > $OUT/logcat-full.txt; grep -i -E "chromium|pxanetwork" $OUT/logcat-full.txt | tail -40 > $OUT/logcat.txt
$ADB emu kill >/dev/null 2>&1
exit $rc
