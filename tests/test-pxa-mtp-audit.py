#!/usr/bin/env python3
"""tools/pxa-mtp-audit.py: local + HTTP-range header audit of the draft (MTP / nextn) head.
Tiny GGUFs (gguf-py), a local Range-capable HTTP server; no network, no GPU.   exit 0 = PASS"""
import http.server, importlib.util, json, os, shutil, sys, tempfile, threading, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gguf-py"))
import numpy as np  # noqa: E402
import gguf  # noqa: E402

_s = importlib.util.spec_from_file_location("audit", os.path.join(HERE, "..", "tools", "pxa-mtp-audit.py"))
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)


def make(path, nextn=None, head=False, arch="qwen35", extra=()):
    w = gguf.GGUFWriter(path, arch)
    w.add_block_count(1); w.add_context_length(256); w.add_embedding_length(64)
    w.add_feed_forward_length(128); w.add_head_count(4); w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5); w.add_file_type(0)
    if nextn is not None:
        w.add_uint32(f"{arch}.nextn_predict_layers", nextn)
    w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(300)])      # a string array the reader must walk
    w.add_array("tokenizer.ggml.scores", [0.0] * 300)                         # a numeric one it must skip
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float32))
    if head:
        w.add_tensor("blk.1.nextn.eh_proj.weight", np.zeros((32, 64), dtype=np.float32))
    for n in extra:
        w.add_tensor(n, np.zeros((32, 64), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    hits = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if not os.path.isfile(p):
            self.send_error(404); return
        data = open(p, "rb").read()
        if rng and rng.startswith("bytes="):
            a, b = rng[6:].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        RangeHandler.hits.append(rng)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="pxa-mtp-audit-")
        j = lambda n: os.path.join(cls.d, n)
        make(j("Good-MTP.gguf"), 1, True)
        make(j("Liar.gguf"), 1, False)
        make(j("NoHead.gguf"), None, False)
        make(j("Claims-MTP.gguf"), None, False)                 # the name says MTP, the file has none
        make(j("HeadNoKv.gguf"), None, True)
        make(j("Stripped-nomtp.gguf"), 1, True)                  # name says no MTP, carries one
        make(j("nemo.gguf"), None, False, arch="nemotron_h")
        make(j("Split-00001-of-00002.gguf"), 1, False)           # the head lives in shard 2
        make(j("Split-00002-of-00002.gguf"), 1, True)
        os.makedirs(j("card"))
        make(j("card/A.gguf"), None, False)
        make(j("card/B.gguf"), 1, True)
        open(j("card/README.md"), "w").write("| `B.gguf` | carries the MTP draft head |\n| `A.gguf` | plain |\n")
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: RangeHandler(*a, directory=cls.d, **k))
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); shutil.rmtree(cls.d, ignore_errors=True)

    def audit(self, where, card="", sib=1):
        f = A.facts_for(where)
        return f, {c for _, c, _ in A.judge(os.path.basename(where), f, card, None, sib)}

    def codes(self, name, **k):
        return self.audit(os.path.join(self.d, name), **k)[1]

    def test_ok_and_plain(self):
        self.assertEqual(self.codes("Good-MTP.gguf"), {"OK"})
        self.assertEqual(self.codes("NoHead.gguf"), {"OK"})

    def test_liar_kv(self):
        self.assertIn("LIAR-KV", self.codes("Liar.gguf"))

    def test_name_claims_mtp_without_head(self):
        self.assertEqual(self.codes("Claims-MTP.gguf"), {"CLAIM-NO-HEAD"})

    def test_head_without_kv_and_no_claim_with_head(self):
        self.assertEqual(self.codes("HeadNoKv.gguf"), {"HEAD-NO-KV"})
        self.assertIn("NOCLAIM-HEAD", self.codes("Stripped-nomtp.gguf"))

    def test_separate_head_arch_is_info(self):
        self.assertEqual(self.codes("nemo.gguf"), {"SEPARATE-HEAD"})

    def test_split_file_head_in_shard_two(self):
        f, c = self.audit(os.path.join(self.d, "Split-00001-of-00002.gguf"))
        self.assertEqual((f["shards"], f["head_tensors"]), (2, 1))
        self.assertEqual(c, {"OK"})

    def test_card_lines_are_per_file(self):
        card = open(os.path.join(self.d, "card/README.md")).read()
        self.assertEqual(self.audit(os.path.join(self.d, "card/A.gguf"), card, 2)[1], {"OK"})     # B's MTP line is not A's claim
        self.assertEqual(self.audit(os.path.join(self.d, "card/B.gguf"), card, 2)[1], {"OK"})
        self.assertIn("CLAIM-NO-HEAD", self.audit(os.path.join(self.d, "card/A.gguf"), "| `A.gguf` | MTP draft head included |\n", 2)[1])
        self.assertIn("CLAIM-NO-HEAD", self.audit(os.path.join(self.d, "NoHead.gguf"), "This model carries the MTP draft head.\n", 1)[1])

    def test_http_range_matches_local_and_reads_little(self):
        RangeHandler.hits.clear()
        for n in ("Good-MTP.gguf", "Liar.gguf", "Split-00001-of-00002.gguf"):
            fl = A.facts_for(os.path.join(self.d, n))
            fh = A.facts_for(f"{self.url}/{n}")
            self.assertEqual({k: fl[k] for k in ("arch", "nextn_kv", "head_tensors", "n_tensors", "shards")},
                             {k: fh[k] for k in ("arch", "nextn_kv", "head_tensors", "n_tensors", "shards")}, n)
        self.assertTrue(all(h and h.startswith("bytes=") for h in RangeHandler.hits))       # only ranges were requested

    def test_missing_shard_and_exit_codes(self):
        self.assertEqual(A.main([os.path.join(self.d, "Good-MTP.gguf")]), 0)
        self.assertEqual(A.main([os.path.join(self.d, "Liar.gguf")]), 1)
        self.assertEqual(A.main([os.path.join(self.d, "HeadNoKv.gguf")]), 0)
        self.assertEqual(A.main(["--strict", os.path.join(self.d, "HeadNoKv.gguf")]), 1)
        self.assertEqual(A.main(["--url", f"{self.url}/Liar.gguf"]), 1)

    def test_preset_claim(self):
        pd = os.path.join(self.d, "presets"); os.makedirs(pd)
        open(os.path.join(pd, "x.env"), "w").write("# NoHead.gguf with --spec-type mtp:n_max=3\n")
        f = A.facts_for(os.path.join(self.d, "NoHead.gguf"))
        self.assertIn("CLAIM-NO-HEAD", {c for _, c, _ in A.judge("NoHead.gguf", f, "", pd)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
