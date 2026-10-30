#!/bin/bash
# `module load pxa/<ver>` works with Environment Modules AND Lmod (packaging/modulefiles/pxa.tcl, staged by
# scripts/make-release-tarball.sh as <package>/modulefiles/pxa/<ver>).
#
# Each module system runs in its own throwaway container (ubuntu:24.04 + environment-modules / lmod; the two
# small images are built once if missing). Inside, a fake package tree (stub bin/, lib/, lib-compat/) gets the
# modulefile, and these are checked:
#   fast     CPU with avx avx2 fma f16c: PATH = <root>/bin:<root>:..., LD_LIBRARY_PATH = <root>/lib first,
#            PXA_ENGINE_DIR = <root>, CUDA_DEVICE_ORDER untouched, `command -v llama-server` = the package's
#   compat   CPU without avx2: <root>/lib-compat FIRST, then <root>/lib; PXA_CPU_LIB_PICKED=compat
#   force    PXA_CPU_LIB=fast on the old CPU -> no lib-compat; PXA_CPU_LIB=compat on the fast CPU -> lib-compat
#   unload   PATH and LD_LIBRARY_PATH come back byte-identical
#   moved    the whole package copied elsewhere -> root follows it; a site MODULEPATH symlink -> root resolved
#   astray   the modulefile copied away from any package -> the load fails with the message
#   whatis   `module whatis pxa` names the version
#
# usage: tests/test-modulefile.sh [modulefile]     exit 0 = PASS, 1 = FAIL, 77 = SKIP (no docker)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MF="${1:-$HERE/../packaging/modulefiles/pxa.tcl}"
MF="$(readlink -f "$MF")"
[ -f "$MF" ] || { echo "no modulefile at $MF"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "SKIP: no docker"; exit 77; }

declare -A PKG=([envmod]="environment-modules tcl" [lmod]="lmod")
declare -A INIT=([envmod]="/usr/share/modules/init/bash" [lmod]="/usr/share/lmod/lmod/init/bash")

inner=$(cat <<'INNER'
set -u
. "$INIT"
fails=0
ok()   { echo "PASS $SYS/$1: $2"; }
bad()  { echo "FAIL $SYS/$1: $2"; fails=$((fails+1)); }
t()    { if [ "$2" = "$3" ]; then ok "$1" "$2"; else bad "$1" "got [$2] want [$3]"; fi; }
mkpkg() {  # mkpkg <root>
  mkdir -p "$1/bin" "$1/lib" "$1/lib-compat" "$1/modulefiles/pxa"
  printf '#!/bin/sh\necho stub\n' > "$1/bin/llama-server"; chmod +x "$1/bin/llama-server"
  : > "$1/lib/libggml.so"; : > "$1/lib-compat/libggml.so"
  printf '#!/bin/sh\n' > "$1/pxa-launch"; chmod +x "$1/pxa-launch"
  cp /mf "$1/modulefiles/pxa/4.0"
}
printf 'processor\t: 0\nflags\t\t: fpu sse sse2 avx avx2 fma f16c bmi2\n' > /tmp/cpu-fast
printf 'processor\t: 0\nflags\t\t: fpu sse sse2 sse4_2 avx\n' > /tmp/cpu-old
R=/opt/pxa-4.0; mkpkg $R
export PATH=/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/usr/local/lib/other
P0="$PATH"; L0="$LD_LIBRARY_PATH"
unset CUDA_DEVICE_ORDER PXA_CPU_LIB PXA_ROOT
module use $R/modulefiles

PXA_CPUINFO=/tmp/cpu-fast module load pxa/4.0 2>/tmp/err
t fast/PATH "$PATH" "$R/bin:$R:$P0"
t fast/LD "$LD_LIBRARY_PATH" "$R/lib:$L0"
t fast/ENGINE "${PXA_ENGINE_DIR:-}" "$R"
t fast/CUDA_ORDER "${CUDA_DEVICE_ORDER:-unset}" "unset"
t fast/which "$(command -v llama-server)" "$R/bin/llama-server"
t fast/quiet "$(cat /tmp/err)" ""
PXA_CPUINFO=/tmp/cpu-fast module unload pxa
t unload/PATH "$PATH" "$P0"; t unload/LD "$LD_LIBRARY_PATH" "$L0"
t unload/ENGINE "${PXA_ENGINE_DIR:-unset}" "unset"

PXA_CPUINFO=/tmp/cpu-old module load pxa/4.0 2>/tmp/err
t compat/LD "$LD_LIBRARY_PATH" "$R/lib-compat:$R/lib:$L0"
t compat/PICKED "${PXA_CPU_LIB_PICKED:-}" "compat"
grep -q 'no avx2 fma f16c' /tmp/err && ok compat/note "$(cat /tmp/err)" || bad compat/note "stderr [$(cat /tmp/err)]"
PXA_CPUINFO=/tmp/cpu-old module unload pxa
t compat-unload/LD "$LD_LIBRARY_PATH" "$L0"; t compat-unload/PICKED "${PXA_CPU_LIB_PICKED:-unset}" "unset"

PXA_CPU_LIB=fast PXA_CPUINFO=/tmp/cpu-old module load pxa/4.0 2>/dev/null
t force-fast/LD "$LD_LIBRARY_PATH" "$R/lib:$L0"
PXA_CPU_LIB=fast PXA_CPUINFO=/tmp/cpu-old module unload pxa
PXA_CPU_LIB=compat PXA_CPUINFO=/tmp/cpu-fast module load pxa/4.0 2>/dev/null
t force-compat/LD "$LD_LIBRARY_PATH" "$R/lib-compat:$R/lib:$L0"
PXA_CPU_LIB=compat PXA_CPUINFO=/tmp/cpu-fast module unload pxa
t force-unload/LD "$LD_LIBRARY_PATH" "$L0"

module unuse $R/modulefiles
R2=/srv/apps/x/pxa; mkdir -p /srv/apps/x; cp -a $R $R2
module use $R2/modulefiles; PXA_CPUINFO=/tmp/cpu-fast module load pxa/4.0 2>/dev/null
t moved/ENGINE "${PXA_ENGINE_DIR:-}" "$R2"; PXA_CPUINFO=/tmp/cpu-fast module unload pxa; module unuse $R2/modulefiles
mkdir -p /etc/site-mods/pxa; ln -s $R2/modulefiles/pxa/4.0 /etc/site-mods/pxa/4.0
module use /etc/site-mods; PXA_CPUINFO=/tmp/cpu-fast module load pxa/4.0 2>/dev/null
t symlink/ENGINE "${PXA_ENGINE_DIR:-}" "$R2"; t symlink/PATH "$PATH" "$R2/bin:$R2:$P0"
w=$(module whatis pxa 2>&1); case "$w" in *"PXA 4.0"*) ok whatis "$(echo "$w" | grep -m1 'PXA 4.0' | tr -s ' ')" ;; *) bad whatis "[$w]" ;; esac
PXA_CPUINFO=/tmp/cpu-fast module unload pxa; module unuse /etc/site-mods

mkdir -p /tmp/lost/pxa; cp /mf /tmp/lost/pxa/4.0; module use /tmp/lost
PXA_CPUINFO=/tmp/cpu-fast module load pxa/4.0 2>/tmp/err
e=$(tr -s ' \n' '  ' < /tmp/err)   # Lmod wraps its message
if [ -z "${PXA_ENGINE_DIR:-}" ] && [[ "$e" == *"no bin/ and lib/ under /tmp"* ]]; then ok astray "load refused, message shown"; else bad astray "ENGINE=${PXA_ENGINE_DIR:-} err=[$(cat /tmp/err)]"; fi
exit $((fails > 0))
INNER
)

rc=0
for SYS in envmod lmod; do
  IMG="pxa-modtest-$SYS:24.04"
  if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends %s >/dev/null && rm -rf /var/lib/apt/lists/*\n' \
      "${PKG[$SYS]}" | docker build -q -t "$IMG" - >/dev/null || { echo "SKIP: could not build $IMG"; exit 77; }
  fi
  docker run --rm --network none -e SYS=$SYS -e INIT="${INIT[$SYS]}" -v "$MF":/mf:ro "$IMG" bash -c "$inner" || rc=1
done
echo "RESULT $([ $rc = 0 ] && echo PASS || echo FAIL)"
exit $rc
