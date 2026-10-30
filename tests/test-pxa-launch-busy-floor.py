#!/usr/bin/env python3
"""R-20 busy-card floor, desktop-aware (RC2.3, todo rc23-busy-floor-display).

nvidia-smi is mocked (the launcher's _run), so this runs anywhere, no GPU:

  1. headless   our box: display_active Disabled, driver model N/A, a container holds 4224 MiB with its
                PID hidden -> card refused (the rule of c68d85fe9d, unchanged); an idle card is free; the
                -q -x process dump is NOT read on a headless box (one small query only).
  2. wddm       the rig's Windows desktop: WDDM, 550 MiB, only C+G programs (dwm.exe, explorer.exe ...)
                with N/A memory -> free, desktop share 550; the same card at 4000 MiB with nothing named
                (over a quarter of 12 GiB + 512) -> refused.
  3. linux      display_active Enabled, Xorg + gnome-shell (G) 430 MiB: 600 MiB used -> free; 1200 MiB
                (770 MiB not the desktop's) -> refused; display Enabled inside a container (no processes
                visible) at 900 MiB -> free.
  4. compute    a real compute program (type C) on the display card -> refused and named, even under the
                floor; a headless card with a listed compute app at 300 MiB -> refused (unchanged).
  5. two cards  card 0 drives the display (free), card 1 is headless with 700 MiB -> only card 1 refused.
  6. plan       plan_and_build-level check through the R-20 site: the refusal text names the reason and
                says how to go on (Allow busy cards / --allow-busy), with no shared/live-box wording.
  7. script     the frozen seat script's shell check with a fake nvidia-smi: a WDDM card at 550 MiB
                passes, a headless card at 700 MiB stops with exit 2.
  8. old-rule   on headless cards card_busy() equals the c68d85fe9d rule on a grid of used/procs.

usage: test-pxa-launch-busy-floor.py [--launcher tools/pxa-launch.py]   exit 0 = PASS
"""
import argparse, importlib.util, os, re, stat, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)


def xml(gpus):
    """gpus = [(uuid, [(pid, type, name, mem_text)])] -> a minimal nvidia-smi -q -x document."""
    out = ["<?xml version=\"1.0\" ?>", "<nvidia_smi_log>"]
    for uuid, procs in gpus:
        out += ["<gpu id=\"00000000:42:00.0\">", f"<uuid>{uuid}</uuid>", "<processes>"]
        for pid, t, name, mem in procs:
            out += ["<process_info>", "<gpu_instance_id>N/A</gpu_instance_id>", f"<pid>{pid}</pid>",
                    f"<type>{t}</type>", f"<process_name>{name}</process_name>", f"<used_memory>{mem}</used_memory>",
                    "</process_info>"]
        out += ["</processes>", "</gpu>"]
    out.append("</nvidia_smi_log>")
    return "\n".join(out)


class FakeSmi:
    def __init__(self, query_gpu, compute_apps="", qx=None):
        self.query_gpu, self.compute_apps, self.qx, self.calls = query_gpu, compute_apps, qx, []

    def __call__(self, cmd, timeout=20):
        self.calls.append(cmd)
        if cmd[:1] != ["nvidia-smi"]:
            return None
        if any(c.startswith("--query-gpu=uuid,display_active") for c in cmd):
            return self.query_gpu
        if any(c.startswith("--query-compute-apps") for c in cmd):
            return self.compute_apps
        if cmd[1:3] == ["-q", "-x"]:
            return self.qx
        return None


WIN_DESKTOP = [("2668", "C+G", r"C:\Windows\System32\dwm.exe", "N/A"),
               ("14580", "C+G", r"C:\Windows\explorer.exe", "N/A"),
               ("17712", "C+G", r"C:\Program Files\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA Overlay.exe", "N/A"),
               ("4", "C+G", "", "N/A")]
WIN_APPS = "\n".join(f"{p}, {n or '[Insufficient Permissions]'}, [N/A], GPU-rig" for p, _t, n, _m in WIN_DESKTOP)


def row(i, used, total=12288, uuid=None, name="NVIDIA GeForce RTX 3080 Ti", cc=86):
    return (i, name, cc, total, used, uuid or f"GPU-{i}")


def run(L, smi, gpus, sel=None):
    L._run = smi
    return L.busy_cards(gpus, sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", default=os.path.join(HERE, "..", "tools", "pxa-launch.py"))
    a = ap.parse_args()
    os.environ.pop("PXA_LAUNCH_FAKE_GPUS", None)
    spec = importlib.util.spec_from_file_location("pxa_launch_busy", a.launcher)
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)

    # 1. headless box, container PIDs hidden
    box = [row(0, 4224, 16384, "GPU-a", "Tesla V100-PCIE-16GB", 70), row(1, 3, 16384, "GPU-b", "Tesla P100-PCIE-16GB", 60)]
    smi = FakeSmi("GPU-a, Disabled, [N/A]\nGPU-b, Disabled, [N/A]", "")
    r = run(L, smi, box)
    check("headless.busy", r[0][0] is True and r[0][2] == 0, r[0])
    check("headless.free", r[1] == (False, "", 0), r[1])
    check("headless.no_qx", not any(c[1:3] == ["-q", "-x"] for c in smi.calls), "no -q -x on a headless box")
    check("headless.513", run(L, FakeSmi("GPU-b, Disabled, [N/A]"), [row(1, 513, uuid="GPU-b")])[1][0] is True, "513 MiB refused")
    check("headless.512", run(L, FakeSmi("GPU-b, Disabled, [N/A]"), [row(1, 512, uuid="GPU-b")])[1][0] is False, "512 MiB free")
    # an old driver without driver_model.current: the first query fails, the fallback answers
    class Old(FakeSmi):
        def __call__(self, cmd, timeout=20):
            if any("driver_model" in c for c in cmd):
                self.calls.append(cmd); return None
            return super().__call__(cmd, timeout)
    o = Old("GPU-a, Disabled", "")
    check("headless.old_driver", run(L, o, [box[0]])[0][0] is True, "fallback query still refuses")

    # 2. Windows WDDM desktop, the rig's numbers
    rig = [row(0, 550, uuid="GPU-rig")]
    smi = FakeSmi("GPU-rig, Disabled, WDDM", WIN_APPS, xml([("GPU-rig", WIN_DESKTOP)]))
    r = run(L, smi, rig)[0]
    check("wddm.550.free", r[0] is False and r[2] == 550, r)
    r = run(L, FakeSmi("GPU-rig, Disabled, WDDM", WIN_APPS, xml([("GPU-rig", WIN_DESKTOP)])), [row(0, 4000, uuid="GPU-rig")])[0]
    check("wddm.4000.busy", r[0] is True and r[2] == 1536 and "2464 MiB" in r[1], r)
    r = run(L, FakeSmi("GPU-rig, Disabled, WDDM", WIN_APPS, xml([("GPU-rig", WIN_DESKTOP)])), [row(0, 2048, uuid="GPU-rig")])[0]
    check("wddm.edge.free", r[0] is False and r[2] == 1536, f"2048 = 1536 desktop cap + 512 floor: {r}")
    r = run(L, FakeSmi("GPU-rig, Disabled, WDDM", WIN_APPS, xml([("GPU-rig", WIN_DESKTOP)])), [row(0, 2049, uuid="GPU-rig")])[0]
    check("wddm.edge.busy", r[0] is True, f"2049 refused: {r}")
    # WSL: the Windows programs are invisible, Xwayland (G, N/A) is all that is listed
    r = run(L, FakeSmi("GPU-rig, Disabled, WDDM", "", xml([("GPU-rig", [("1", "G", "/Xwayland", "N/A")])])),
            [row(0, 568, uuid="GPU-rig")])[0]
    check("wsl.568.free", r[0] is False and r[2] == 568, r)
    # -q -x unreadable: the type of each program is unknown -> share up to the cap, N/A apps are not counted as compute
    r = run(L, FakeSmi("GPU-rig, Disabled, WDDM", WIN_APPS, None), rig)[0]
    check("wddm.noxml.free", r[0] is False and r[2] == 550, r)

    # 3. Linux desktop, display_active Enabled
    lx = [("2101", "G", "/usr/lib/xorg/Xorg", "250 MiB"), ("2300", "G", "/usr/bin/gnome-shell", "180 MiB")]
    for used, want in ((600, False), (942, False), (943, True), (1200, True)):
        r = run(L, FakeSmi("GPU-l, Enabled, [N/A]", "", xml([("GPU-l", lx)])), [row(0, used, 8192, "GPU-l")])[0]
        check(f"linux.{used}", r[0] is want and r[2] == min(used, 430), r)
    r = run(L, FakeSmi("GPU-l, Enabled, [N/A]", "", xml([("GPU-l", [])])), [row(0, 900, 12288, "GPU-l")])[0]
    check("linux.container.900", r[0] is False and r[2] == 900, r)

    # 4. real compute use
    comp = WIN_DESKTOP + [("9001", "C", r"D:\pxa\bin\llama-server.exe", "N/A")]
    r = run(L, FakeSmi("GPU-rig, Disabled, WDDM", WIN_APPS, xml([("GPU-rig", comp)])), [row(0, 400, uuid="GPU-rig")])[0]
    check("compute.wddm", r[0] is True and "llama-server.exe (pid 9001)" in r[1], r)
    r = run(L, FakeSmi("GPU-l, Enabled, [N/A]", "4242, /usr/bin/python3, 300, GPU-l",
                       xml([("GPU-l", lx + [("4242", "C", "/usr/bin/python3", "300 MiB")])])), [row(0, 730, 8192, "GPU-l")])[0]
    check("compute.linux_display", r[0] is True and "python3 (pid 4242, 300 MiB)" in r[1], r)
    r = run(L, FakeSmi("GPU-a, Disabled, [N/A]", "4242, /usr/bin/python3, 300, GPU-a"), [row(0, 300, 16384, "GPU-a")])[0]
    check("compute.headless", r[0] is True and "python3 (pid 4242, 300 MiB)" in r[1], r)
    r = run(L, FakeSmi("GPU-a, Disabled, [N/A]", "2123725, [Not Found], 1332, GPU-a"), [row(0, 1341, 16384, "GPU-a")])[0]
    check("compute.other_container", r[0] is True and "a program it cannot name (pid 2123725, 1332 MiB)" in r[1], r)

    # REVIEW of 4de1cebf8d, one test per finding
    # F1: a C+G program holding more than 512 MiB on its own is compute (Blender Cycles as C+G, 6000 MiB)
    r = run(L, FakeSmi("GPU-l, Enabled, [N/A]", "", xml([("GPU-l", lx + [("777", "C+G", "/usr/bin/blender", "6000 MiB")])])),
            [row(0, 6430, 8192, "GPU-l")])[0]
    check("review.f1.cg_big_is_compute", r[0] is True and "blender (pid 777, 6000 MiB)" in r[1], r)
    cgs = [(str(900 + k), "C+G", f"/usr/bin/app{k}", "400 MiB") for k in range(5)]   # 5 x 400 = 2000 MiB, each under 512
    r = run(L, FakeSmi("GPU-l, Enabled, [N/A]", "", xml([("GPU-l", cgs)])), [row(0, 2100, 8192, "GPU-l")])[0]
    check("review.f1.graphics_over_cap", r[0] is True and r[2] == 1536 and "564 MiB" in r[1], r)
    # F2: nothing listed (WSL / Docker Desktop on Windows) on a 24 GB card: fixed 1536 MiB share, not card/4
    for used, want in ((2048, False), (4000, True)):
        r = run(L, FakeSmi("GPU-w, Disabled, WDDM", "", xml([("GPU-w", [])])), [row(0, used, 24576, "GPU-w", "NVIDIA GeForce RTX 4090", 89)])[0]
        check(f"review.f2.nolist24g.{used}", r[0] is want and r[2] == min(used, 1536), r)
    # F4: MPS types count as compute
    for t in ("M", "M+C"):
        r = run(L, FakeSmi("GPU-l, Enabled, [N/A]", "", xml([("GPU-l", lx + [("55", t, "/usr/bin/nvidia-cuda-mps-server", "300 MiB")])])),
                [row(0, 730, 8192, "GPU-l")])[0]
        check(f"review.f4.mps.{t}", r[0] is True and "nvidia-cuda-mps-server" in r[1], r)
    # any other type with a C (not C+G) is compute; an unknown empty type is not named but its memory still counts
    r = run(L, FakeSmi("GPU-l, Enabled, [N/A]", "", xml([("GPU-l", lx + [("56", "C+X", "/usr/bin/odd", "10 MiB")])])),
            [row(0, 450, 8192, "GPU-l")])[0]
    check("review.f4.other_c", r[0] is True and "odd" in r[1], r)

    # 5. two cards: display + headless
    two = [row(0, 600, 8192, "GPU-l"), row(1, 700, 16384, "GPU-h", "Tesla P100-PCIE-16GB", 60)]
    r = run(L, FakeSmi("GPU-l, Enabled, [N/A]\nGPU-h, Disabled, [N/A]", "", xml([("GPU-l", lx), ("GPU-h", [])])), two)
    check("two.display_free", r[0][0] is False and r[0][2] == 430, r[0])
    check("two.headless_busy", r[1][0] is True and r[1][2] == 0, r[1])
    r = run(L, FakeSmi("GPU-l, Enabled, [N/A]\nGPU-h, Disabled, [N/A]", "", xml([("GPU-l", lx), ("GPU-h", [])])), two, [two[0]])
    check("two.sel_only", list(r) == [0], r)

    # 6. the refusal wording through the R-20 template
    plan = L.Plan()
    busy, why, _d = run(L, FakeSmi("GPU-a, Disabled, [N/A]", ""), [box[0]])[0]
    plan.refuse("R-20", i=0, mib=4224, procs=why)
    t = plan.refusals[0][1]
    check("text.howto", "Allow busy cards" in t and "--allow-busy" in t and "4224 MiB" in t, t)
    check("text.no_box_words", not re.search(r"SHARED|LIVE box|shared, live", t, re.I), "no shared/live-box wording")
    check("text.no_money", not re.search(r"\b(paid|sell|buy|price)\b", t + why, re.I), "no paid/sell/buy/price")
    check("text.no_env_switch", not re.search(r"\bPXA_[A-Z_]+", t), "no PXA_* switch in the user text")

    # 7. the frozen seat script's shell check
    tmp = tempfile.mkdtemp(prefix="busyfloor-")
    try:
        bind = os.path.join(tmp, "bin"); os.makedirs(bind)
        fake = os.path.join(bind, "nvidia-smi")
        with open(fake, "w") as f:
            f.write("#!/bin/sh\ncase \"$*\" in\n"
                    "  *memory.used*) echo \"$FAKE_USED\";;\n  *display_active*) echo \"$FAKE_DISP\";;\n"
                    "  *memory.total*) echo \"$FAKE_TOTAL\";;\n  *query-compute-apps*) [ -n \"$FAKE_APPS\" ] && printf '%s\\n' \"$FAKE_APPS\";;\nesac\nexit 0\n")
        os.chmod(fake, 0o755)
        p = L.write_serve_script("busyfloor-test", ["/bin/true"], {}, "0", tmp)
        sc = p if isinstance(p, str) and os.path.exists(p) else next(
            (os.path.join(dp, fn) for dp, _d2, fns in os.walk(tmp) for fn in fns if fn.endswith(".sh")), None)
        body = open(sc).read() if sc else ""
        check("script.written", "limit=512" in body, sc or "no script")
        cut = body.split("# Card numbers are nvidia-smi's")[0] + "\necho PASSED\n"
        chk = os.path.join(tmp, "chk.sh")
        open(chk, "w").write(cut)
        wapps = "2668, [N/A]\n14580, [N/A]"
        for tag, used, disp, apps, ok in (("wddm550", "550", "Disabled, WDDM", wapps, True),
                                          ("headless700", "700", "Disabled, [N/A]", "", False),
                                          ("headless300_app", "300", "Disabled, [N/A]", "4242, 300", True),   # old rule: memory only
                                          ("linux900", "900", "Enabled, [N/A]", "", True),
                                          ("wddm2048", "2048", "Disabled, WDDM", wapps, True),
                                          ("wddm4000", "4000", "Disabled, WDDM", wapps, False),
                                          ("review.f3.linux1500_python", "1500", "Enabled, [N/A]", "4242, 1000", False)):
            env = dict(os.environ, PATH=bind + os.pathsep + os.environ.get("PATH", ""), FAKE_USED=used,
                       FAKE_DISP=disp, FAKE_TOTAL="8192", FAKE_APPS=apps)
            rr = subprocess.run(["sh", chk], capture_output=True, text=True, env=env)
            good = ("PASSED" in rr.stdout and rr.returncode == 0) if ok else (rr.returncode == 2)
            check(f"script.{tag}", good, f"rc={rr.returncode} {rr.stderr.strip()[:120]}")
    finally:
        subprocess.run(["rm", "-rf", tmp])

    # 9. review (e): Control caches display_info 30 s and never runs two reads at once
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.launcher)))
    import threading, time, types
    import pxa_control.core as core
    calls, inflight, peak = [], [0], [0]
    mx = threading.Lock()

    def slow(rows):
        with mx:
            inflight[0] += 1; peak[0] = max(peak[0], inflight[0])
        calls.append(time.time()); time.sleep(0.3)
        with mx:
            inflight[0] -= 1
        return {0: {"display": True}}
    stub = types.SimpleNamespace(L=types.SimpleNamespace(display_info=slow), DISPLAY_INFO_TTL_S=30.0,
                                 _disp_lock=threading.Lock(), _disp_cache=None)
    rows = [row(0, 550, uuid="GPU-rig")]
    th = [threading.Thread(target=core.App.display_info_cached, args=(stub, rows)) for _ in range(6)]
    [t.start() for t in th]; [t.join() for t in th]
    check("review.e.first_once", len(calls) == 1, f"{len(calls)} reads for 6 concurrent first polls")
    for _ in range(10):
        core.App.display_info_cached(stub, rows)
    check("review.e.ttl", len(calls) == 1, f"{len(calls)} reads after 10 polls inside 30 s")
    stub.DISPLAY_INFO_TTL_S = 0.0
    res = []
    th = [threading.Thread(target=lambda: res.append(core.App.display_info_cached(stub, rows))) for _ in range(6)]
    [t.start() for t in th]; [t.join() for t in th]
    check("review.e.no_two_in_flight", peak[0] == 1 and len(res) == 6 and all(r == {0: {"display": True}} for r in res),
          f"peak {peak[0]} read(s) in flight, {len(calls) - 1} read(s) for 6 concurrent stale polls; all answered")

    # 10. review 11c: a REAL App (lock made in __init__), N threads make the very first call at once:
    #     exactly one read in flight, no exception (the old lazy lock could race -> 'release unlocked lock')
    os.environ.setdefault("PXA_CONTROL_CONFIG_DIR", tempfile.mkdtemp(prefix="busyfloor-cfg-"))
    calls[:] = []; peak[0] = 0
    app = core.App(L, port=0, models_dirs=[tempfile.mkdtemp(prefix="busyfloor-models-")])
    app.L = types.SimpleNamespace(display_info=slow)
    errs, res = [], []

    def first():
        try:
            res.append(app.display_info_cached(rows))
        except Exception as e:   # noqa: BLE001
            errs.append(repr(e))
    th = [threading.Thread(target=first) for _ in range(12)]
    [t.start() for t in th]; [t.join() for t in th]
    check("review.11c.first_call_race", not errs and peak[0] == 1 and len(calls) == 1 and len(res) == 12,
          f"errors {errs[:2]}, peak {peak[0]} in flight, {len(calls)} read(s), {len(res)} answers")

    # 8. headless == the old rule on a grid
    bad = 0
    for used in (0, 5, 511, 512, 513, 550, 4224):
        for procs in ([], [("1", "x", 3)]):
            old = bool(procs) or used > 512
            new = L.card_busy(row(0, used), procs, None)[0]
            new2 = L.card_busy(row(0, used), procs, {"display": False, "wddm": False, "active": False, "procs": None})[0]
            bad += (old != new) + (old != new2)
    check("oldrule.grid", bad == 0, f"{bad} mismatches over 28 cases")

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
