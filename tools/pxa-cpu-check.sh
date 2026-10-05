#!/bin/sh
# Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
#
# pxa-cpu-check.sh - SOURCED by run-server.sh, pxa-launch and pxa-entrypoint, never run on its own.
#
# The fast library in lib/ is built with AVX, AVX2, FMA and F16C. A CPU without them (a pre-2013 Xeon, an
# AVX-only Sandy or Ivy Bridge, a VM that hides the flags) dies with "Illegal instruction" the moment the
# library loads, before any model is read. The package therefore carries a second libggml.so built for plain
# x86-64 in lib-compat/ (same GPU code; only the CPU-side work is slower), and this function picks it.
#
#   pxa_cpu_pick <dir of lib-compat>
#
# What it reads: the first "flags" line of /proc/cpuinfo.
# What it does:  CPU has avx avx2 fma f16c  -> nothing (the fast lib/ is used).
#                CPU lacks one of them      -> puts <dir> FIRST on LD_LIBRARY_PATH, prints ONE plain line to
#                                              stderr, exports PXA_CPU_LIB_PICKED=compat.
#                no flags line (not x86, or /proc not mounted) -> nothing: it cannot tell, so it does not guess.
# Returns:       0, or 1 when the CPU needs the compat library and <dir> is not there (it says so in one line;
#                the caller exits, because going on would end in "Illegal instruction").
# Overrides:     PXA_CPU_LIB=fast|compat|auto (default auto) forces a choice.
#                PXA_CPUINFO=<file> reads that file instead of /proc/cpuinfo (the test hook).
#                PXA_CPU_NOTE_DONE=1 (set by this function) keeps the child launchers from repeating the line.

pxa_cpu_missing() {
    _f=${PXA_CPUINFO:-/proc/cpuinfo}
    [ -r "$_f" ] || return 0
    _line=$(grep -m1 '^flags[[:space:]]*:' "$_f" 2>/dev/null) || return 0
    [ -n "$_line" ] || return 0
    _miss=
    for _want in avx avx2 fma f16c; do
        case " $_line " in
            *" $_want "*) ;;
            *) _miss="$_miss $_want" ;;
        esac
    done
    # leading blank trimmed; empty = nothing is missing (or the CPU could not be read)
    printf '%s' "${_miss# }"
}

pxa_cpu_pick() {
    _dir=${1:-}
    case "${PXA_CPU_LIB:-auto}" in
        fast) return 0 ;;
        compat) _miss="forced by PXA_CPU_LIB=compat" ;;
        *)
            _miss=$(pxa_cpu_missing)
            [ -n "$_miss" ] || return 0
            _miss="no ${_miss}"
            ;;
    esac
    if [ -n "$_dir" ] && [ -f "$_dir/libggml.so" ]; then
        LD_LIBRARY_PATH="$_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export LD_LIBRARY_PATH
        PXA_CPU_LIB_PICKED=compat
        export PXA_CPU_LIB_PICKED
        if [ -z "${PXA_CPU_NOTE_DONE:-}" ]; then
            echo "pxa: this CPU is older than the fast library needs ($_miss), so the compatibility library in lib-compat is used. Graphics-card speed is the same; only the work done on the CPU is slower." >&2
            PXA_CPU_NOTE_DONE=1
            export PXA_CPU_NOTE_DONE
        fi
        return 0
    fi
    echo "pxa: this CPU is older than the fast library needs ($_miss) and this install has no lib-compat folder, so the program would stop with 'Illegal instruction'. Use the full release package (it carries lib-compat)." >&2
    return 1
}
