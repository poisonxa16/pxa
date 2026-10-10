#!/usr/bin/env python3
"""Adaptive expert map.

The engine reads a csv next to the GGUF: tensor,expert,count. One pass of
that file is the curated map. This builder asks the library for each
calibration step, records the speed, and stops when the library says to
stop. The fastest map is the one that is kept.

It does not change the model file. It does not write the csv inside a
version directory the updater swaps. The csv sits next to the GGUF, or in
the directory the caller names. A learned copy is a separate file. The
library decides when that file is used. No library: the curated file.
"""
import argparse
import csv
import ctypes
import json
import os
import shutil
import sys
import urllib.request

_PXQN = None
_PXQN_TRIED = False


def _pxqn():
    global _PXQN, _PXQN_TRIED
    if _PXQN_TRIED:
        return _PXQN
    _PXQN_TRIED = True
    path = os.environ.get("PXA_PXQN_LIB") or "libggml-pxqn.so"
    try:
        _PXQN = ctypes.CDLL(path)
    except OSError:
        _PXQN = None
    return _PXQN


def _sym(name, restype, argtypes):
    lib = _pxqn()
    if lib is None:
        return None
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return None
    fn.restype = restype
    fn.argtypes = argtypes
    return fn

DISCLAIMER = (
    "This map is a starting guess of which experts stay on the card. "
    "Each pass adds a short calibration and the speed is measured again. "
    "When the speed stops rising, the builder stops and keeps the fastest map. "
    "After that, the running server can still move experts as you chat. "
    "It does not change the model file, and it does not touch your other models."
)


def tensor_name(layer):
    return "blk.%d.ffn_down_exps.weight" % layer


def read_csv(path):
    rows = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "tensor" not in r.fieldnames:
            raise ValueError("counts file needs tensor,expert,count columns")
        for rec in r:
            key = (rec["tensor"], int(rec["expert"]))
            rows[key] = rows.get(key, 0) + int(float(rec["count"]))
    if not rows:
        raise ValueError("counts file has no rows")
    return rows


def write_csv(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("tensor", "expert", "count"))
        for (tensor, expert), count in sorted(rows.items()):
            w.writerow((tensor, expert, int(count)))
    os.replace(tmp, path)


def rows_from_log(obj):
    counts = obj.get("counts") or []
    rows = {}
    for il, row in enumerate(counts):
        name = tensor_name(il)
        for expert, count in enumerate(row):
            if int(count) > 0:
                rows[(name, expert)] = int(count)
    if not rows:
        raise ValueError("expert log has no counts")
    return rows


def merge_rows(*parts):
    out = {}
    for part in parts:
        for key, count in part.items():
            out[key] = out.get(key, 0) + int(count)
    return out


def _calib_step(which, index):
    """One step from the library. None past the end. Raises when the library is absent."""
    fn = _sym("ggml_pxqn_xcache_calib_step", ctypes.c_int,
              [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
               ctypes.c_char_p, ctypes.c_size_t,
               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)])
    if fn is None:
        raise RuntimeError("calibration needs libggml-pxqn")
    kind = ctypes.create_string_buffer(32)
    prompt = ctypes.create_string_buffer(512)
    n_predict = ctypes.c_int(0)
    temperature = ctypes.c_double(0)
    rc = fn(int(which), int(index), kind, 32, prompt, 512,
            ctypes.byref(n_predict), ctypes.byref(temperature))
    if rc < 0:
        raise RuntimeError("the library rejected the calibration step")
    if rc == 0:
        return None
    return kind.value.decode(), prompt.value.decode(), int(n_predict.value), float(temperature.value)


def consider(history, prose, code, hit):
    """Ask the library whether this measurement is kept. No library: refuse."""
    fn = _sym("ggml_pxqn_xcache_calib_consider", ctypes.c_int,
              [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double,
               ctypes.POINTER(ctypes.c_int)])
    if fn is None:
        raise RuntimeError("the stop decision needs libggml-pxqn")
    n = len(history)
    hist = (ctypes.c_double * (max(n, 1) * 2))()
    for i, row in enumerate(history):
        hist[2 * i] = float(row["prose"])
        hist[2 * i + 1] = float(row["code"])
    best = ctypes.c_int(-1)
    rc = fn(hist if n else None, n, float(prose), float(code), ctypes.byref(best))
    if rc < 0:
        raise RuntimeError("the library rejected the measurement")
    if rc == 0:
        history.append({"prose": float(prose), "code": float(code),
                        "hit": None if hit is None else float(hit)})
    return rc == 1, int(best.value)


def curated_start(model_path, curated_csv, dest):
    """Copy the curated map to dest. Never overwrite the curated file."""
    if os.path.abspath(curated_csv) == os.path.abspath(dest):
        raise ValueError("the working map must not be the curated file")
    if not model_path.endswith(".gguf"):
        raise ValueError("model path must be a .gguf")
    shutil.copyfile(curated_csv, dest)
    return dest


def beside_model(model_path):
    return model_path + ".expert-counts.csv"


def counts_fetch_target(model_url, local_path):
    """Where a shard fetch saves the curated expert map.

    Returns (url, dest), or None when the model URL has no file name.
    dest is the local model path plus '.expert-counts.csv': the file the
    server opens, which for a split GGUF is shard 1. url is the model URL
    with its own file name plus that suffix, query and fragment removed.
    A fetch must not replace dest when that file is already there.
    """
    if not model_url or not local_path:
        return None
    base = model_url.split("#", 1)[0].split("?", 1)[0]
    slash = base.rfind("/")
    if slash < 0 or slash + 1 >= len(base):
        return None
    remote = base[slash + 1 :]
    if not remote or "/" in remote:
        return None
    url = base[: slash + 1] + remote + ".expert-counts.csv"
    return url, local_path + ".expert-counts.csv"


def counts_fetch_replaces_existing():
    """A shard fetch leaves a counts file that is already beside the model alone."""
    return False


def _post_json(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def calibrate(base_url, log_dir, timeout=600):
    """Send the calibration steps the library names. The server must have been
    started with PXA_EXPERT_LOG pointed at log_dir. Returns the merged routing rows."""
    before = set(os.listdir(log_dir)) if os.path.isdir(log_dir) else set()
    index = 0
    while True:
        step = _calib_step(0, index)
        if step is None:
            break
        _kind, prompt, n_predict, temperature = step
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature, "cache_prompt": False}
        _post_json(base_url.rstrip("/") + "/completion", body, timeout)
        index += 1
    rows = {}
    for name in sorted(os.listdir(log_dir)):
        if name in before or not name.endswith(".json") or name == "index.jsonl":
            continue
        with open(os.path.join(log_dir, name)) as f:
            rows = merge_rows(rows, rows_from_log(json.load(f)))
    if not rows:
        raise RuntimeError("calibration wrote no expert log")
    return rows


def hit_rate(base_url, timeout=10):
    props = _get_json(base_url.rstrip("/") + "/props", timeout)
    xc = props.get("pxa_xcache") or {}
    hits = float(xc.get("hits") or 0)
    misses = float(xc.get("misses") or 0)
    if hits + misses <= 0:
        return None
    return hits / (hits + misses)


def decode_once(base_url, kind, timeout=600):
    index = 0
    step = None
    while True:
        got = _calib_step(1, index)
        if got is None:
            break
        if got[0] == kind:
            step = got
            break
        index += 1
    if step is None:
        raise ValueError("unknown calibration kind")
    _kind, prompt, n_predict, temperature = step
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature, "cache_prompt": False}
    data = _post_json(base_url.rstrip("/") + "/completion", body, timeout)
    timings = data.get("timings") or {}
    tps = timings.get("predicted_per_second")
    if tps is None:
        raise RuntimeError("the server did not report decode speed")
    return float(tps)


def learned_beside(model_path):
    return model_path + ".expert-counts.learned.csv"


def cache_learned(model_path, cache_dir=None):
    base = os.path.basename(model_path)
    if not cache_dir:
        cache_dir = os.environ.get("PXA_CACHE_DIR") or ""
        if not cache_dir:
            xdg = os.environ.get("XDG_CACHE_HOME") or ""
            if xdg:
                cache_dir = os.path.join(xdg, "pxa")
            else:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "pxa")
    return os.path.join(cache_dir, base + ".expert-counts.learned.csv")


def write_speed(path, prose, code):
    """Record a measurement beside a counts file. The library writes it."""
    parent = os.path.dirname(os.path.abspath(path + ".speed"))
    os.makedirs(parent, exist_ok=True)
    fn = _sym("ggml_pxqn_xcache_speed_write", ctypes.c_int,
              [ctypes.c_char_p, ctypes.c_double, ctypes.c_double])
    if fn is None:
        raise RuntimeError("recording a measurement needs libggml-pxqn")
    if fn(os.fsencode(path), float(prose), float(code)) != 0:
        raise OSError("could not record the measurement")


def pick_counts(curated, learned, explicit=False):
    """Which counts file a boot should open. The library decides.

    No library: the curated path, never the learned one.
    explicit is PXA_XCACHE_COUNTS: that path is not swapped.
    """
    fn = _sym("ggml_pxqn_xcache_counts_pick", ctypes.c_int,
              [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
               ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)])
    if fn is None:
        return curated, "curated"
    learned_mean = ctypes.c_double(-1)
    curated_mean = ctypes.c_double(-1)
    use = fn(os.fsencode(curated) if curated else None,
             os.fsencode(learned) if learned else None,
             1 if explicit else 0,
             ctypes.byref(learned_mean), ctypes.byref(curated_mean))
    if use == 1 and learned:
        return learned, "learned"
    return curated, "curated"


def read_state(path):
    sessions = 0
    learning = 0
    st = (path or "") + ".state"
    if path and os.path.isfile(st):
        with open(st) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                if parts[0] == "sessions":
                    sessions = int(parts[1])
                elif parts[0] == "learning":
                    learning = int(parts[1])
    return sessions, 1 if learning else 0


def reset_learned(paths):
    """Delete learned csv, state, and speed. Never a curated counts file."""
    for path in paths:
        if not path:
            continue
        if not path.endswith(".expert-counts.learned.csv"):
            raise ValueError("refusing to reset a file that is not a learned counts file")
        for extra in ("", ".state", ".speed"):
            victim = path + extra
            if os.path.isfile(victim):
                os.remove(victim)


def cache_warning(cache_dir=None, in_docker=None):
    """Plain sentence when learned counts would not survive. Empty when they would."""
    if cache_dir is None:
        cache_dir = os.environ.get("PXA_CACHE_DIR")
        if not cache_dir:
            xdg = os.environ.get("XDG_CACHE_HOME")
            cache_dir = os.path.join(xdg, "pxa") if xdg else os.path.join(os.path.expanduser("~"), ".cache", "pxa")
    parent = cache_dir if os.path.isdir(cache_dir) else (os.path.dirname(cache_dir) or ".")
    writable = os.path.isdir(parent) and os.access(parent, os.W_OK)
    if not writable:
        return "The cache directory %s is not writable. Learned counts will not be kept." % cache_dir
    if in_docker is None:
        in_docker = os.path.exists("/.dockerenv")
    if not in_docker:
        return ""
    try:
        same = os.stat(parent).st_dev == os.stat("/").st_dev
    except OSError:
        return ""
    if not same:
        return ""
    return ("Learned counts in %s will be deleted when this container is removed. "
            "Mount a volume and set PXA_CACHE_DIR to it, for example "
            "-v pxa-cache:%s -e PXA_CACHE_DIR=%s." % (cache_dir, cache_dir, cache_dir))


def status_dict(model_path=None, working=None, history=None, best=None, stopped=False):
    sessions = 0
    learning = False
    learned = None
    if model_path:
        for cand in (learned_beside(model_path), cache_learned(model_path)):
            if os.path.isfile(cand) or os.path.isfile(cand + ".state"):
                learned = cand
                sessions, learning_n = read_state(cand)
                learning = bool(learning_n)
                break
    return {
        "disclaimer": DISCLAIMER,
        "model": model_path,
        "working": working,
        "passes": history or [],
        "best": best,
        "stopped": stopped,
        "online_adapt": "PXA_XCACHE_ADAPT=1",
        "learning": learning,
        "sessions": sessions,
        "learned": learned,
        "cache_warning": cache_warning(),
    }


def main(argv):
    p = argparse.ArgumentParser(description="adaptive expert map")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("disclaimer")
    m = sub.add_parser("selftest")
    m.add_argument("--tmp", default="")
    a = p.parse_args(argv)
    if a.cmd == "disclaimer":
        print(DISCLAIMER)
        return 0
    if a.cmd == "selftest":
        return _selftest(a.tmp or None)
    return 2


def _selftest(tmp):
    import tempfile
    root = tmp or tempfile.mkdtemp(prefix="pxa-expert-map-")
    os.makedirs(root, exist_ok=True)
    curated = os.path.join(root, "curated.csv")
    work = os.path.join(root, "work.expert-counts.csv")
    write_csv(curated, {(tensor_name(0), 1): 10, (tensor_name(0), 2): 1})
    model = os.path.join(root, "model.gguf")
    open(model, "w").close()
    curated_start(model, curated, work)
    assert read_csv(work)[(tensor_name(0), 1)] == 10
    try:
        curated_start(model, curated, curated)
        raise SystemExit("curated overwrite was allowed")
    except ValueError:
        pass
    log = {"counts": [[0, 4, 0], [1, 0, 2]]}
    merged = merge_rows(read_csv(work), rows_from_log(log))
    assert merged[(tensor_name(0), 1)] == 14
    assert merged[(tensor_name(1), 2)] == 2
    write_csv(work, merged)
    text = open(work).read().splitlines()
    assert text[0] == "tensor,expert,count"
    learned = learned_beside(model)
    write_csv(learned, {(tensor_name(0), 1): 3, (tensor_name(0), 2): 9})
    # no library in this check: the curated file, never the learned one
    chosen, why = pick_counts(curated, learned)
    assert chosen == curated and why == "curated", (chosen, why)
    chosen, why = pick_counts(curated, learned, explicit=True)
    assert chosen == curated and why == "curated", (chosen, why)
    os.remove(curated)
    chosen, why = pick_counts("", learned)
    assert chosen == "" and why == "curated", (chosen, why)
    try:
        reset_learned([curated])
        raise SystemExit("curated reset was allowed")
    except ValueError:
        pass
    assert os.path.isfile(learned)
    reset_learned([learned])
    assert not os.path.isfile(learned)
    assert not os.path.isfile(learned + ".speed")
    assert os.path.isfile(work)
    st = learned + ".state"
    with open(st, "w") as f:
        f.write("sessions 2\nlearning 1\n")
    sessions, learning = read_state(learned)
    assert sessions == 2 and learning == 1
    info = status_dict(model)
    assert info["learning"] is True and info["sessions"] == 2
    assert "flat_tps" not in info
    reset_learned([learned])
    assert not os.path.isfile(st)
    assert "not writable" in cache_warning("/no/such/pxa-cache-dir", in_docker=False)
    assert cache_warning(".", in_docker=False) == ""
    print("PASS")
    print(status_dict(model, work, [], None, False)["disclaimer"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
