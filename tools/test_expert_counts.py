#!/usr/bin/env python3
"""Expert routing counts (v3.1): the stage a mixture-of-experts model gets, the file it writes, and the two things
Control must never do - ask an old encoder for a flag it will refuse the run over, and write a GGUF to disk without the
counts file the engine reads beside it.

No GPU, no encoder, no network: the encoder command line is a tiny fake, the GGUF is written here, and the counts
format is checked against the curated reference file when it is on this machine.

    python3 -m pytest tools/ -k expert_counts -q
    python3 tools/test_expert_counts.py
"""
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pxa_encode_adapter as AD            # noqa: E402
import pxa_encode_plan as PL               # noqa: E402
import pxa_expert_map as EM                # noqa: E402

# The curated counts file is a build-machine artifact and is NOT in the tree, so the path comes
# from the environment only: PXA_TEST_COUNTS_CSV=<path>.  Unset (the normal case for anyone who
# did not run the calibration), the reference check reports "skipped (no reference file)".
REFERENCE_CSV = os.environ.get("PXA_TEST_COUNTS_CSV", "")
REFERENCE_MD5 = "04db6ebb423ed2c509b196c03bf72334"


# ---- a GGUF header written here, so a MoE model needs no 20 GB file ------------------------------------------------
def _kv_str(k, v):
    kb, vb = k.encode(), v.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_u32(k, v):
    kb = k.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", int(v))


def _kv_arr_str(k, vals):
    kb = k.encode()
    out = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(vals))
    for v in vals:
        vb = v.encode()
        out += struct.pack("<Q", len(vb)) + vb
    return out


def write_gguf(path, kvs, n_tensors=0):
    """A GGUF whose metadata is exactly `kvs` (a list of already-encoded pairs) and whose tensor block is empty."""
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) + struct.pack("<Q", len(kvs)))
        for kv in kvs:
            f.write(kv)


def moe_gguf(path, experts=128):
    # a tokenizer array first: the reader must walk past a value it does not want to reach the key it does
    write_gguf(path, [_kv_arr_str("tokenizer.ggml.tokens", ["a", "b", "c"]),
                      _kv_str("general.architecture", "qwen3moe"),
                      _kv_u32("qwen3moe.expert_count", experts),
                      _kv_u32("qwen3moe.block_count", 48)])
    return path


def dense_gguf(path):
    write_gguf(path, [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 32)])
    return path


# ---- a fake encoder that writes a counts file ---------------------------------------------------------------------
def fake_encoder(path, rows=4, body=True, header=AD.COUNTS_HEADER, rc=0):
    """A `pxqe counts` that writes a real counts file at --out, or a broken one, or nothing and exits `rc`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n")
        if rc == 0 and body:
            f.write('out=""\n')
            f.write('while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done\n')
            f.write('printf "%s\\n" "%s" > "$out"\n' % (header, ""))
            for i in range(rows):
                f.write('printf "blk.0.ffn_gate_exps.weight,%d,%d\\n" >> "$out"\n' % (i, 50 + i))
        if rc != 0:
            f.write("echo 'counts: no room on the device' >&2\n")
        f.write("exit %d\n" % rc)
    os.chmod(path, 0o755)
    return path


# ---- the argv -----------------------------------------------------------------------------------------------------
def test_counts_argv_shape():
    a = AD.argv_counts("/usr/local/bin/pxqe", "/m/x.gguf", "/m/calib.txt", "/m/x.gguf.expert-counts.csv")
    assert a == ["/usr/local/bin/pxqe", "counts", "--src", "/m/x.gguf", "--calib", "/m/calib.txt", "--out", "/m/x.gguf.expert-counts.csv"]
    # a .py wrapper runs under this interpreter, like every other verb
    assert AD.argv_counts("/w/pxqe.py", "a", "b", "c")[:2] == [sys.executable, "/w/pxqe.py"]


def test_counts_file_path_is_the_one_the_engine_reads():
    assert AD.counts_file("/m/x.gguf") == "/m/x.gguf.expert-counts.csv"
    assert AD.counts_file("/m/x.gguf") == PL.counts_file("/m/x.gguf")


def test_make_argv_counts_flags():
    base = AD.argv_make("pxqe", "org/m", "pxqn4", "/o/f.gguf", "/w")
    assert "--counts" not in base and "--no-counts" not in base and "--allow-no-counts" not in base
    assert "--counts" in AD.argv_make("pxqe", "org/m", "pxqn4", "/o/f.gguf", "/w", counts=True)
    assert "--no-counts" in AD.argv_make("pxqe", "org/m", "pxqn4", "/o/f.gguf", "/w", counts=False)
    assert "--allow-no-counts" in AD.argv_make("pxqe", "org/m", "pxqn4", "/o/f.gguf", "/w", allow_no_counts=True)
    # the same argv resumes the job, so the counts choice is part of what a started job is
    a = AD.argv_make("pxqe", "org/m", "pxqn4", "/o/f.gguf", "/w", counts=True, lock="supporters")
    b = AD.argv_make("pxqe", "org/m", "pxqn4", "/o/f.gguf", "/w", counts=True, lock="supporters")
    assert a == b and "--counts" in a


# ---- the progress lines -------------------------------------------------------------------------------------------
def _line(o):
    import json
    return "@pxqe " + json.dumps(o)


def test_a_counts_stage_in_the_plan_and_in_the_progress():
    ev = AD.parse_make_line(_line({"event": "plan", "stages": [{"name": "encode", "weight": 4}, {"name": "counts", "weight": 1},
                                                                 {"name": "verify", "weight": 1}], "tier": "pxqn4"}))
    assert [s["name"] for s in ev["stages"]] == ["encode", "counts", "verify"]
    assert AD.MAKE_STAGE_ID["counts"] == "counts" and AD.JOB_STAGE_MAKE["counts"] == "counts"
    st = AD.parse_make_line(_line({"event": "stage", "stage": "counts", "state": "progress", "percent": 40, "overall_percent": 91,
                                   "message": "24576 rows"}))
    assert st == {"event": "stage", "stage": "counts", "state": "progress", "percent": 0.4, "eta_s": None, "overall": 0.91, "message": "24576 rows"}
    assert AD.parse_make_line(_line({"event": "stage", "stage": "counts", "state": "done"}))["state"] == "done"
    assert AD.parse_make_line(_line({"event": "stage", "stage": "counts", "state": "skipped"}))["state"] == "skipped"
    # a stage the page does not know is still dropped, not half-parsed
    assert AD.parse_make_line(_line({"event": "stage", "stage": "nonsense", "state": "done"})) is None


def test_a_failed_counts_stage_is_explained_as_a_warning_not_a_lost_run():
    ex = AD.explain_make_failure(1, {"code": "counts-failed", "message": "counting stopped: the device is full", "resumable": True})
    assert ex["code"] == "counts-failed" and ex["resumable"] is True
    assert "the model file itself is finished" in ex["hint"].lower() or "Press Resume" in ex["hint"]
    assert ex["message"].endswith(".") and "PXQE_KEY" not in ex["message"] and "/" not in ex["message"]
    # the encoder's own code with no message still gets Control's sentence, not a traceback
    ex2 = AD.explain_make_failure(1, {"code": "counts-failed"})
    assert ex2["code"] == "counts-failed" and ex2["message"].endswith(".") and ex2["hint"]


# ---- reading a counts file ----------------------------------------------------------------------------------------
def test_counts_check_accepts_a_well_formed_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.gguf.expert-counts.csv")
        with open(p, "w") as f:
            f.write("tensor,expert,count\nblk.0.ffn_gate_exps.weight,0,51\nblk.0.ffn_gate_exps.weight,1,56\nblk.0.ffn_down_exps.weight,0,7\n")
        r = AD.counts_check(p)
        assert r["ok"] is True and r["rows"] == 3 and r["tensors"] == 2 and r["why"] == ""


def test_counts_check_refuses_a_file_the_engine_cannot_use():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.csv")
        cases = [("", "empty"), ("expert,count\n0,1\n", "header"),
                 ("tensor,expert,count\n", "no expert rows"),
                 ("tensor,expert,count\nblk.0.w,x,3\n", "int"),
                 ("tensor,expert,count\nblk.0.w,0,-3\n", "int"),
                 ("tensor,expert,count\n,0,3\n", "int")]
        for body, why in cases:
            with open(p, "w") as f:
                f.write(body)
            r = AD.counts_check(p)
            assert r["ok"] is False, (body, why)
            assert why.split()[0] in r["why"] or r["why"], (body, r["why"])
        assert AD.counts_check(os.path.join(d, "gone.csv"))["ok"] is False


def test_counts_check_reports_rows_and_tensors_for_the_log():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.csv")
        with open(p, "w") as f:
            f.write("tensor,expert,count\n" + "".join("blk.%d.ffn_gate_exps.weight,%d,1\n" % (i, e) for i in range(3) for e in range(8)))
        r = AD.counts_check(p)
        assert r["ok"] and r["rows"] == 24 and r["tensors"] == 3


def test_the_curated_reference_file_still_has_its_header_and_shape():
    """Spec: the format a new file is checked against. Skips when the reference is not on this machine."""
    if not REFERENCE_CSV or not os.path.isfile(REFERENCE_CSV):
        return "skipped (no reference file)"
    import hashlib
    h = hashlib.md5(open(REFERENCE_CSV, "rb").read()).hexdigest()
    assert h == REFERENCE_MD5, h
    r = AD.counts_check(REFERENCE_CSV)
    # 24576 lines of body above the header, 48 layers of `ffn_down_exps.weight`, 512 experts each
    assert r["ok"] is True and r["rows"] == 24576 and r["tensors"] == 48


# ---- running the verb, and never leaving a half-written CSV -------------------------------------------------------
def test_counts_run_writes_the_file_only_when_it_is_usable():
    with tempfile.TemporaryDirectory() as d:
        cli = fake_encoder(os.path.join(d, "pxqe"), rows=5)
        out = os.path.join(d, "x.gguf.expert-counts.csv")
        r = AD.counts_run(cli, os.path.join(d, "x.gguf"), os.path.join(d, "calib.txt"), out)
        assert r["ok"] is True and r["rows"] == 5
        assert os.path.isfile(out) and AD.counts_check(out)["ok"] is True
        assert not os.path.exists(AD.counts_temp(out)), "the temp name is left behind"


def test_a_failed_or_broken_counts_run_leaves_no_csv_beside_the_model():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "x.gguf.expert-counts.csv")
        # the encoder exits non-zero
        r = AD.counts_run(fake_encoder(os.path.join(d, "bad"), rc=3), os.path.join(d, "x.gguf"), "c.txt", out)
        assert r["ok"] is False and not os.path.exists(out) and not os.path.exists(AD.counts_temp(out))
        assert "device" in r["why"] or r["code"] == "counts"
        # the encoder "succeeds" but writes a file the engine cannot use: still no CSV
        r = AD.counts_run(fake_encoder(os.path.join(d, "junk"), header="hello,world"), os.path.join(d, "x.gguf"), "c.txt", out)
        assert r["ok"] is False and not os.path.exists(out) and not os.path.exists(AD.counts_temp(out))
        # no rows at all
        r = AD.counts_run(fake_encoder(os.path.join(d, "empty"), body=False), os.path.join(d, "x.gguf"), "c.txt", out)
        assert r["ok"] is False and not os.path.exists(out)
        # a missing encoder is a sentence, not a traceback
        r = AD.counts_run(os.path.join(d, "nope"), os.path.join(d, "x.gguf"), "c.txt", out)
        assert r["ok"] is False and r["why"] and not os.path.exists(out)


def test_an_existing_csv_is_never_overwritten_without_asking():
    with tempfile.TemporaryDirectory() as d:
        cli = fake_encoder(os.path.join(d, "pxqe"), rows=2)
        out = os.path.join(d, "x.gguf.expert-counts.csv")
        with open(out, "w") as f:
            f.write("tensor,expert,count\nkeep.me,0,1\n")
        r = AD.counts_run(cli, "x.gguf", "c.txt", out)
        assert r["ok"] is False and "already" in r["why"]
        assert open(out).read().endswith("keep.me,0,1\n")
        r = AD.counts_run(cli, "x.gguf", "c.txt", out, overwrite=True)
        assert r["ok"] is True and "keep.me" not in open(out).read()


# ---- which models get the stage -----------------------------------------------------------------------------------
def test_a_gguf_with_experts_needs_counts_and_a_dense_one_does_not():
    with tempfile.TemporaryDirectory() as d:
        m = moe_gguf(os.path.join(d, "moe.gguf"), 128)
        assert PL.gguf_kvs(m, {"general.architecture"})["general.architecture"] == "qwen3moe"
        assert PL.expert_count(m) == 128 and PL.counts_needed(m) is True
        dense = dense_gguf(os.path.join(d, "dense.gguf"))
        assert PL.expert_count(dense) == 0 and PL.counts_needed(dense) is False
        # one expert is a degenerate case the spec counts as expert-bearing (expert_count > 0)
        assert PL.counts_needed(moe_gguf(os.path.join(d, "one.gguf"), 1)) is True
        # not a GGUF, a missing file, or a value that is not a count: dense, never a crash
        assert PL.expert_count(os.path.join(d, "dense.gguf")) == 0
        assert PL.expert_count(os.path.join(d, "gone.gguf")) == 0
        junk = os.path.join(d, "junk.gguf")
        with open(junk, "wb") as f:
            f.write(b"not a gguf at all")
        assert PL.expert_count(junk) == 0


def test_the_inspection_dict_and_a_config_dict_both_answer():
    assert PL.expert_count({"kind": "gguf", "moe": True, "experts": 256}) == 256
    assert PL.expert_count({"experts": 0, "moe": False}) == 0
    assert PL.expert_count({"num_experts": 64}) == 64 and PL.expert_count({"n_routed_experts": 8}) == 8
    assert PL.expert_count({"num_experts_per_tok": 8}) == 0, "that is how many a token uses, not how many exist"
    assert PL.expert_count(None) == 0 and PL.expert_count({"experts": True}) == 0


def test_the_stage_list_gains_counts_only_for_a_model_with_experts():
    moe = {"experts": 128}
    dense = {"experts": 0}
    make = ["download", "convert", "reference", "skeleton", "encode", "verify"]
    assert PL.with_counts_stage(make, moe) == ["download", "convert", "reference", "skeleton", "encode", "counts", "verify"]
    assert PL.with_counts_stage(make, dense) == make
    assert PL.with_counts_stage(make, dense) is not make, "the caller's list is never mutated in place"
    # idempotent, and a list with no verify still gets the stage at the end
    once = PL.with_counts_stage(make, moe)
    assert PL.with_counts_stage(once, moe) == once
    assert PL.with_counts_stage(["encode"], moe) == ["encode", "counts"]


# ---- only an encoder that knows the stage is asked for it ---------------------------------------------------------
def test_a_shard_fetch_saves_the_counts_file_beside_the_model():
    """The download that fetches shards also names <model>.expert-counts.csv next to the file the server opens."""
    one = "https://huggingface.co/org/repo/resolve/main/Swift-1.5-Qwen3.8-Flash-Next-PXQN-32GB.gguf"
    local = "/models/Swift-1.5-Qwen3.8-Flash-Next-PXQN-32GB.gguf"
    url, dest = EM.counts_fetch_target(one, local)
    assert url == one + ".expert-counts.csv"
    assert dest == local + ".expert-counts.csv"
    # a split GGUF is opened at shard 1, so the map is that shard's name, not shard 2 or 3
    shard = "https://huggingface.co/org/repo/resolve/main/Qwen3.8-Flash-Next-Uncensored-PXQN-00001-of-00003.gguf?download=true#frag"
    spath = "/models/Qwen3.8-Flash-Next-Uncensored-PXQN-00001-of-00003.gguf"
    url, dest = EM.counts_fetch_target(shard, spath)
    assert url.endswith("/Qwen3.8-Flash-Next-Uncensored-PXQN-00001-of-00003.gguf.expert-counts.csv")
    assert "?" not in url and "#" not in url
    assert dest == spath + ".expert-counts.csv"
    assert "00002-of-00003" not in url and "00002-of-00003" not in dest
    # the local path is where the server looks, even when the cache name differs from the URL name
    url, dest = EM.counts_fetch_target(one, "/cache/renamed.gguf")
    assert url.endswith("/Swift-1.5-Qwen3.8-Flash-Next-PXQN-32GB.gguf.expert-counts.csv")
    assert dest == "/cache/renamed.gguf.expert-counts.csv"
    assert EM.counts_fetch_target("", local) is None
    assert EM.counts_fetch_target(one, "") is None
    assert EM.counts_fetch_target("https://huggingface.co/org/repo/resolve/main/", local) is None
    assert EM.counts_fetch_replaces_existing() is False


def test_the_engine_shard_fetch_uses_that_rule_and_a_miss_still_loads():
    """common.cpp, no build: the optional map is requested after the shards, and a miss does not abort the load."""
    cpp = os.path.join(os.path.dirname(HERE), "common", "common.cpp")
    text = open(cpp, encoding="utf-8").read()
    start = text.find("static bool pxa_counts_fetch_target")
    assert start > 0
    body = text[start:text.find("static bool llama_download_file", start)]
    assert '.expert-counts.csv"' in body
    assert "local_path + \".expert-counts.csv\"" in body
    # an existing sidecar returns before any delete
    dl = text.find("static bool llama_download_file")
    early = text[dl:text.find("previous metadata file found", dl)]
    assert "if (optional && file_exists)" in early and "return true" in early
    load = text.find("struct llama_model * llama_load_model_from_url")
    tail = text[load:text.find("struct llama_model * llama_load_model_from_hf", load)]
    ask = tail.find("llama_download_file(counts_url, counts_dest, hf_token, true)")
    assert ask > 0
    # the shards are fetched first; a failed counts download does not return NULL before the model load
    assert tail.find("llama_model_load_from_file") > ask
    assert "return NULL" not in tail[ask:]


def test_only_an_encoder_that_lists_the_counts_stage_is_asked():
    assert AD.supports_counts({"stages": [{"name": "encode"}, {"name": "counts", "ready": True}]}) is True
    assert AD.supports_counts({"stages": [{"name": "encode"}, {"name": "verify"}]}) is False
    assert AD.supports_counts({}) is False and AD.supports_counts(None) is False
    assert AD.supports_counts({"stages": ["counts"]}) is False, "a bare string is not a stage record"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            r = fn()
            print("ok   %s%s" % (fn.__name__, (" (%s)" % r) if isinstance(r, str) else ""))
        except Exception:                                   # noqa: BLE001
            bad += 1
            print("FAIL %s" % fn.__name__)
            traceback.print_exc()
    print("\n%d passed, %d failed" % (len(fns) - bad, bad))
    sys.exit(1 if bad else 0)
