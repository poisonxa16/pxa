#!/bin/bash
# The ONE Linux download: pxa-<ver>-linux-x86_64.run = a small sh header + one tar.gz holding the compiled PXA Control
# (packaging/control/build-linux.sh), the engine (bin/ lib/ lib-compat/ of the release) and the icon.
# Running it (double-click "Run" or ./pxa-....run) verifies the payload's sha256, unpacks it ONCE into
# ~/.local/share/pxa/<ver> (the install.sh layout, `current` symlink), adds PXA Control to the applications menu and
# starts it: the browser opens signed in. Later starts (menu, or the .run again) skip the unpack.
#   make-linux-run.sh <pxa-control binary> <release dir with bin/ lib/ [lib-compat/] VERSION> <icon.png> <out.run>
set -euo pipefail
BIN="$1"; REL="$2"; ICON="$3"; OUT="$4"
VER="$(sed -n 's/^tag:[[:space:]]*//p' "$REL/VERSION" | head -1)"; VER="${VER:-$(head -1 "$REL/VERSION" | tr -cd 'A-Za-z0-9._-')}"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/pxa/engine"
cp -a "$BIN" "$T/pxa/pxa-control"
for d in bin lib lib-compat; do [ -d "$REL/$d" ] && cp -a "$REL/$d" "$T/pxa/engine/"; done
cp -a "$REL/VERSION" "$T/pxa/VERSION"
cp -a "$ICON" "$T/pxa/pxa-control.png"
if command -v pigz >/dev/null; then tar -C "$T" -cf - pxa | pigz -9 > "$T/payload.tgz"; else tar -C "$T" -czf "$T/payload.tgz" pxa; fi
SHA="$(sha256sum "$T/payload.tgz" | cut -d' ' -f1)"
cat > "$T/header.sh" <<'HDR'
#!/bin/sh
# PXA @VER@ for Linux x86_64 - one file. Run it: it unpacks once into ~/.local/share/pxa/@VER@, adds PXA Control to
# your applications menu and opens PXA Control in your browser. No Python, no pip, no CUDA toolkit needed (the NVIDIA
# driver is). Options go to PXA Control (pxa-control --help).
set -eu
VER="@VER@"; SHA="@SHA@"; SKIP=@SKIP@
glibc=$(ldd --version 2>/dev/null | sed -n '1s/.* \([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p')
case "$glibc" in 2.[0-9]|2.[12][0-9]|2.3[0-4]) echo "PXA needs glibc 2.35 or newer (Ubuntu 22.04+, Debian 12+, RHEL 9+); this system has $glibc." >&2; exit 1;; esac
DEST="${PXA_HOME:-$HOME/.local/share/pxa}"; DIR="$DEST/$VER"
if [ ! -x "$DIR/pxa-control" ] || [ "$(cat "$DIR/.payload-sha256" 2>/dev/null)" != "$SHA" ]; then
  echo "PXA $VER: checking and unpacking into $DIR (once) ..."
  mkdir -p "$DEST"; tmp="$DEST/.unpack-$$"; rm -rf "$tmp"; mkdir -p "$tmp"
  got=$(tail -c +"$SKIP" "$0" | sha256sum | cut -d' ' -f1)
  [ "$got" = "$SHA" ] || { echo "PXA: this download is damaged (checksum mismatch). Download it again." >&2; rm -rf "$tmp"; exit 1; }
  tail -c +"$SKIP" "$0" | tar -xzf - -C "$tmp"
  rm -rf "$DIR"; mv "$tmp/pxa" "$DIR"; rm -rf "$tmp"
  echo "$SHA" > "$DIR/.payload-sha256"
  ln -sfn "$VER" "$DEST/current"
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/pxa-control.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=PXA Control
Comment=Run AI models on your GPUs
Exec="$DEST/current/pxa-control"
Icon=$DEST/current/pxa-control.png
Terminal=false
Categories=Development;Utility;
EOF
fi
exec "$DIR/pxa-control" "$@"
HDR
# SKIP = header bytes + 1 (tail -c +N is 1-based); the placeholder is padded so the header length does not move
sed -i "s/@VER@/$VER/g; s/@SHA@/$SHA/" "$T/header.sh"
printf '\n#__PAYLOAD__\n' >> "$T/header.sh"
sed -i "s/@SKIP@/SKIP_PLACEHOLDER_0000000000/" "$T/header.sh"
LEN=$(stat -c %s "$T/header.sh")
SKIPV=$(printf '%-27s' "$((LEN + 1))")
sed -i "s/SKIP_PLACEHOLDER_0000000000/$SKIPV/" "$T/header.sh"
[ "$(stat -c %s "$T/header.sh")" = "$LEN" ] || { echo "header length moved" >&2; exit 1; }
cat "$T/header.sh" "$T/payload.tgz" > "$OUT"
chmod +x "$OUT"
printf '{"out": "%s", "bytes": %s, "payload_sha256": "%s", "version": "%s"}\n' "$OUT" "$(stat -c %s "$OUT")" "$SHA" "$VER"
