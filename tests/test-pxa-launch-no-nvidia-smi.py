#!/usr/bin/env python3
"""pxa-launch must find the GPUs without nvidia-smi (public pxa #4: containers / LXC that pass the
driver through but not the nvidia-smi binary), and --gpus must work when nothing detects at all.

Runs anywhere (no GPU needed): nvidia-smi is masked from PATH, NVML is switched off with the test
hook, and the /proc/driver/nvidia/gpus fallback reads a fake directory.

  1. procfs   two fake cards listed out of bus order -> indexes follow the PCI bus id (the order
              nvidia-smi and NVML use), capability from the name table, memory from "NNGB".
  2. given    no detector answers, --gpus 1,0 -> the plan runs with those two cards (not exit 2),
              stderr says what was assumed; PXA_LAUNCH_GPU_MEM_MIB / _CC set the assumptions.
  3. refuse   no detector answers, no --gpus -> exit 2 with the new message naming --gpus.
  4. nvml     on a box with the NVIDIA driver: the NVML table equals nvidia-smi's on index, name,
              capability, total memory and UUID (SKIP where there is no driver).

usage: test-pxa-launch-no-nvidia-smi.py [--launcher tools/pxa-launch.py]   exit 0 = PASS
"""
import argparse, importlib.util, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)


def masked_path(tmp):
    """A PATH that has everything on the current PATH except nvidia-smi."""
    d = os.path.join(tmp, "bin")
    os.makedirs(d)
    for p in os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep):
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f == "nvidia-smi" or os.path.exists(os.path.join(d, f)):
                continue
            try:
                os.symlink(os.path.join(p, f), os.path.join(d, f))
            except OSError:
                pass
    return d


def fake_proc(tmp):
    root = os.path.join(tmp, "gpus")
    cards = {"0000:84:00.0": ("Tesla P100-PCIE-16GB", "GPU-bbbb"),
             "0000:05:00.0": ("Tesla V100-PCIE-32GB", "GPU-aaaa")}
    for bus, (model, uuid) in cards.items():
        os.makedirs(os.path.join(root, bus))
        with open(os.path.join(root, bus, "information"), "w") as f:
            f.write(f"Model: \t\t {model}\nIRQ:   \t\t 205\nGPU UUID: \t {uuid}\nBus Type: \t PCIe\n"
                    f"Bus Location: \t {bus}\nDevice Minor: \t 0\n")
    return root


def tiny_gguf(tmp):
    sys.path.insert(0, os.path.join(HERE, "..", "gguf-py"))
    import numpy as np
    import gguf
    p = os.path.join(tmp, "tiny.gguf")
    w = gguf.GGUFWriter(p, "llama")
    w.add_block_count(1); w.add_context_length(256); w.add_embedding_length(64)
    w.add_feed_forward_length(128); w.add_head_count(4); w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5); w.add_file_type(0)
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", default=os.path.join(HERE, "..", "tools", "pxa-launch.py"))
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="pxa-launch-nosmi-")
    real_path = os.environ.get("PATH", "/usr/bin:/bin")
    try:
        path = masked_path(tmp)
        base = {"PATH": path, "HOME": tmp, "PXA_LAUNCH_NO_NVML": "1"}
        if shutil.which("nvidia-smi", path=path):
            print("HARNESS nvidia-smi still on the masked PATH"); return 2

        # 1. procfs parser, in-process
        os.environ.update(base); os.environ["PXA_LAUNCH_PROC_GPUS"] = fake_proc(tmp)
        os.environ.pop("PXA_LAUNCH_FAKE_GPUS", None)
        spec = importlib.util.spec_from_file_location("pxa_launch", a.launcher)
        L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
        rows, err = L.gpu_table()
        want = [(0, "Tesla V100-PCIE-32GB", 70, 32768, 0, "GPU-aaaa"),
                (1, "Tesla P100-PCIE-16GB", 60, 16384, 0, "GPU-bbbb")]
        check("procfs", rows == want and getattr(L, "GPU_SOURCE", None) == "/proc/driver/nvidia/gpus",
              f"rows={rows} err={err} source={getattr(L, 'GPU_SOURCE', None)}")

        model = tiny_gguf(tmp)
        env = dict(base, PXA_LAUNCH_NO_PROC="1")

        def launch(*args, extra=None):
            r = subprocess.run([sys.executable, a.launcher, "--explain", "--model", model] + list(args),
                               env=dict(env, **(extra or {})), capture_output=True, text=True, timeout=300)
            return r.returncode, r.stdout, r.stderr

        # 2. --gpus with nothing detected
        rc, out, errt = launch("--gpus", "1,0")
        check("given/runs", "cannot detect GPUs" in errt and "using --gpus 1,0 as given" in errt
              and "GPU 0 (not detected)" in out and "GPU 1 (not detected)" in out,
              f"rc={rc} stderr tail={errt.strip().splitlines()[-1:]}")
        rc2, out2, errt2 = launch("--gpus", "0", extra={"PXA_LAUNCH_GPU_MEM_MIB": "24576", "PXA_LAUNCH_GPU_CC": "70"})
        check("given/overrides", "sm_70, 24576 MiB per card assumed" in errt2 and "sm_70" in out2,
              f"rc={rc2} stderr={[l for l in errt2.splitlines() if 'assumed' in l]}")

        # 3. nothing detected and no --gpus -> refuse, naming --gpus
        rc3, _, errt3 = launch()
        check("refuse", rc3 == 2 and "Pass --gpus <indexes>" in errt3, f"rc={rc3} stderr={errt3.strip()[:160]}")

        # 4. NVML vs nvidia-smi on a real driver
        os.environ["PATH"] = real_path
        if shutil.which("nvidia-smi") and not hasattr(L, "_gpus_nvml"):
            check("nvml", False, "launcher has no NVML reader")
        elif shutil.which("nvidia-smi"):
            smi_rows, e1 = L._gpus_nvidia_smi()
            nvml_rows, e2 = L._gpus_nvml()
            key = lambda rs: [(r[0], r[1], r[2], r[3], r[5]) for r in (rs or [])]
            check("nvml", bool(smi_rows) and key(smi_rows) == key(nvml_rows),
                  f"{len(smi_rows or [])} smi rows vs {len(nvml_rows or [])} nvml rows ({e1 or ''}{e2 or ''})")
        else:
            print("SKIP nvml: no nvidia-smi on this host to compare against")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT", "PASS" if not fails else "FAIL " + ",".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
