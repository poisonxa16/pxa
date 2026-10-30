#%Module1.0
## PXA modulefile -- `module load pxa/<version>` (Environment Modules 3.2+/4/5 and Lmod read this same file).
##
## Shipped inside the release package as <root>/modulefiles/pxa/<version>. It finds <root> from its own path,
## so the package can live anywhere: `module use <root>/modulefiles` then `module load pxa`. A site that keeps
## its modulefiles elsewhere can symlink this file there (the link is followed) or set PXA_ROOT before loading.
##
## What it sets:
##   PATH             <root>/bin (llama-server, llama-cli, ...) and <root> (pxa, pxa-launch, run-server.sh)
##   LD_LIBRARY_PATH  <root>/lib; <root>/lib-compat goes first when this CPU lacks AVX, AVX2, FMA or F16C
##                    (the fast lib/ would stop with "Illegal instruction"). PXA_CPU_LIB=fast|compat forces it.
##   PXA_ENGINE_DIR   <root> (where pxa-launch finds the engine)
## It does not set CUDA_DEVICE_ORDER or CUDA_VISIBLE_DEVICES: the scheduler / your job owns the cards.

proc pxa_root {} {
    global env ModulesCurrentModulefile
    if {[info exists env(PXA_ROOT)] && $env(PXA_ROOT) ne ""} {
        return [file normalize $env(PXA_ROOT)]
    }
    set f [file normalize $ModulesCurrentModulefile]
    # follow a symlink chain (a site MODULEPATH linking into the package)
    for {set i 0} {$i < 16 && [file type $f] eq "link"} {incr i} {
        set l [file readlink $f]
        if {[file pathtype $l] ne "absolute"} { set l [file join [file dirname $f] $l] }
        set f [file normalize $l]
    }
    # <root>/modulefiles/pxa/<version>
    return [file dirname [file dirname [file dirname $f]]]
}

# the CPU features the fast lib/ is built for, missing on this machine ("" = none missing or unreadable)
proc pxa_cpu_missing {} {
    global env
    set f /proc/cpuinfo
    if {[info exists env(PXA_CPUINFO)] && $env(PXA_CPUINFO) ne ""} { set f $env(PXA_CPUINFO) }
    if {[catch {set fh [open $f r]}]} { return "" }
    set flags ""
    while {[gets $fh line] >= 0} {
        if {[regexp {^flags\s*:\s*(.*)$} $line -> flags]} { break }
    }
    close $fh
    if {$flags eq ""} { return "" }
    set miss {}
    foreach want {avx avx2 fma f16c} {
        if {[lsearch -exact [split $flags " \t"] $want] < 0} { lappend miss $want }
    }
    return [join $miss " "]
}

set root    [pxa_root]
set version [file tail $ModulesCurrentModulefile]

proc ModulesHelp {} {
    global root version
    puts stderr "PXA $version -- llama-server with the PXA speed work, the pxa launcher and PXA Control."
    puts stderr "  pxa                      opens PXA Control (or: pxa --model FILE.gguf --gpus 0,1)"
    puts stderr "  run-server.sh ARGS       llama-server with this package's libraries"
    puts stderr "  llama-server, llama-cli  the engine binaries (bin/)"
    puts stderr "Package: $root"
    puts stderr "PXA_CPU_LIB=fast|compat (set before loading) forces the CPU library choice."
}

module-whatis "PXA $version: LLM inference engine (llama-server), pxa launcher, PXA Control"
conflict pxa

if {![file isdirectory [file join $root bin]] || ![file isdirectory [file join $root lib]]} {
    error "pxa: no bin/ and lib/ under $root -- this modulefile must stay at <package>/modulefiles/pxa/<version> (or set PXA_ROOT)"
}

prepend-path PATH            $root
prepend-path PATH            [file join $root bin]
prepend-path LD_LIBRARY_PATH [file join $root lib]
setenv       PXA_ENGINE_DIR  $root

set pick auto
if {[info exists env(PXA_CPU_LIB)]} { set pick $env(PXA_CPU_LIB) }
set why ""
if {$pick eq "compat"} {
    set why "forced by PXA_CPU_LIB=compat"
} elseif {$pick ne "fast"} {
    set m [pxa_cpu_missing]
    if {$m ne ""} { set why "no $m" }
}
if {$why ne ""} {
    set compat [file join $root lib-compat]
    if {[file exists [file join $compat libggml.so]]} {
        # prepended after lib/, so it comes FIRST
        prepend-path LD_LIBRARY_PATH $compat
        setenv PXA_CPU_LIB_PICKED compat
        if {[module-info mode load]} {
            puts stderr "pxa: this CPU is older than the fast library needs ($why), so the compatibility library in lib-compat is used."
        }
    } elseif {[module-info mode load]} {
        puts stderr "pxa: this CPU is older than the fast library needs ($why) and this package has no lib-compat folder: the programs would stop with 'Illegal instruction'. Use the full release package."
    }
}
