#!/usr/bin/env python3
# test-release-scan-closedlib.py -- scripts/release-scan.py check 6 (closed-lib, owner order 2026-11-01): a public source push must
# not carry a speed mechanism added after the public baseline. Synthetic git fixtures in temp dirs only:
#   bad      a dummy __global__ kernel in a speed path + a new PXA_FOO getenv in a core file -> exit 2, both named file:line
#   newcu    a new .cu file -> exit 2 (new-cuda-file)
#   hook     a change to the versioned hook header only -> exit 0
#   marker   a file that carries the PXA_CLOSED_HOOK marker (a plain fallback) -> exit 0
#   docs     docs + README + a comment-only edit in a speed path -> exit 0
#   tests    a kernel and a lever inside tests/ -> exit 0
#   report   --report on bad -> exit 0 and the same hits listed
#   allow    the allowlist (path / sym, reason, approver) silences hits; a malformed line stops the check (exit 3)
#   dir      an exported directory (no git) of bad -> exit 2; a closed path present in that directory -> exit 2
#   closed   a kernel added in a closed path (scripts/pxa-closed-files.txt) on a git ref -> exit 0 (the export omits it)
#   hunk     a hunk entry (hunk:<path>:<sha256 of the hunk's stripped code lines>) silences that one speed-path hunk only:
#            a second, unrelated hunk in the same file stays a hit; the same hunk with one line changed is a hit again;
#            a malformed hunk entry stops the check (exit 3)
import hashlib, json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(ROOT, "scripts", "release-scan.py")
GENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.invalid", GIT_CONFIG_NOSYSTEM="1", HOME=tempfile.gettempdir())

BASE_FILES = {
    "ggml/src/ggml-cuda/foo.cu": "#include \"common.cuh\"\n\n__global__ void k_old(float * x) { x[0] = 1.0f; }\n\n"
                                 "void foo_launch(float * x) { k_old<<<1, 1>>>(x); }\n",
    "src/llama.cpp": "#include <cstdlib>\n\nstatic int old_lever() {\n    const char * e = getenv(\"PXA_OLD\");\n"
                     "    return e ? atoi(e) : 0;\n}\n\nint llama_main_loop() {\n    return old_lever();\n}\n",
    "ggml/src/ggml-pxqn-api.h": "// the closed library's table\n#define GGML_PXQN_LIB_VERSION 1u\n"
                                "struct ggml_pxqn_lib_api {\n    unsigned version;\n    int (*mul_mat)(void *);\n};\n",
    "ggml/src/ggml-cuda/pxa/fallback.cuh": "// PXA_CLOSED_HOOK: plain fallback for the closed verify GEMV\n"
                                           "static inline int verify_fallback(int n) {\n    return n;\n}\n",
    "docs/LEVERS.md": "# Levers\n\n- PXA_OLD: an old lever\n",
    "README.md": "# engine\n",
    "scripts/pxa-closed-files.txt": "# closed paths\nggml/src/ggml-cuda/pxa/closed.cu\nsrc/closed-dir\n",
    "tests/test-foo.cpp": "int main() { return 0; }\n",
}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo] + list(args), check=True, capture_output=True, text=True, env=GENV).stdout


def write(repo, files):
    for rel, text in files.items():
        p = os.path.join(repo, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)


def branch(repo, name, files):
    git(repo, "checkout", "-q", "-b", name, "base")
    write(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "base")


def scan(repo, args, allow, out):
    cmd = [sys.executable, "-I", SCAN, "--closedlib", "base", "--repo", repo, "--allow", allow, "--json", out] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    js = {}
    if os.path.exists(out):
        with open(out) as f:
            js = json.load(f)
        os.unlink(out)
    return p.returncode, js, p.stdout + p.stderr


def has(js, cat, path, needle=None):
    for h in js.get("hits", []):
        if h.get("category") == cat and h.get("path") == path and (needle is None or needle in h.get("reason", "")):
            return True
    return False


def main():
    fails = 0
    t = tempfile.mkdtemp(prefix="rs-closedlib-")
    try:
        repo = os.path.join(t, "repo")
        os.makedirs(repo)
        git(repo, "init", "-q", "-b", "base")
        write(repo, BASE_FILES)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "baseline")
        foo = BASE_FILES["ggml/src/ggml-cuda/foo.cu"]
        llama = BASE_FILES["src/llama.cpp"]
        branch(repo, "bad", {
            "ggml/src/ggml-cuda/foo.cu": foo + "\n__global__ void k_dummy(float * x) { x[1] = 2.0f; }\n",
            "src/llama.cpp": llama.replace("int llama_main_loop() {\n",
                                           "int llama_main_loop() {\n    if (getenv(\"PXA_FOO\")) { return 7; }\n"),
        })
        branch(repo, "newcu", {"ggml/src/ggml-cuda/bar.cu": "// a new translation unit\nvoid bar() {}\n"})
        branch(repo, "hook", {"ggml/src/ggml-pxqn-api.h": BASE_FILES["ggml/src/ggml-pxqn-api.h"].replace(
            "VERSION 1u", "VERSION 2u").replace("    int (*mul_mat)(void *);\n",
                                                "    int (*mul_mat)(void *);\n    int (*verify_gemv)(void *, int);\n")})
        branch(repo, "marker", {"ggml/src/ggml-cuda/pxa/fallback.cuh": BASE_FILES["ggml/src/ggml-cuda/pxa/fallback.cuh"]
                                + "static inline int verify_fallback2(int n) {\n    return n + 1;\n}\n"})
        branch(repo, "docs", {"docs/LEVERS.md": BASE_FILES["docs/LEVERS.md"] + "- more words\n",
                              "README.md": "# engine\n\nNew words about PXA_OLD.\n",
                              "docs/new-page.md": "# A page\n",
                              "ggml/src/ggml-cuda/foo.cu": foo.replace("__global__ void k_old",
                                                                       "// the old kernel\n__global__ void k_old")})
        branch(repo, "tests", {"tests/test-foo.cpp": "#include <cstdlib>\nint main() { return getenv(\"PXA_TEST_ONLY\") ? 1 : 0; }\n",
                               "tests/test-kern.cu": "__global__ void k_test(int * x) { x[0] = 0; }\nint main() { return 0; }\n"})
        branch(repo, "closed", {"ggml/src/ggml-cuda/pxa/closed.cu": "__global__ void k_closed(float * x) { x[0] = 3.0f; }\n",
                                "src/closed-dir/spec.cpp": "#include <cstdlib>\nint f() { return getenv(\"PXA_CLOSED_LEVER\") ? 1 : 0; }\n"})

        hook_inc = '#include "pxqn-ext-call.cuh"\n'
        hook_call = ("    const bool ext_done = pxqn_ext_gemv(x, 1);\n"
                     "    if (!ext_done) {\n        k_old<<<1, 1>>>(x);\n    }\n")
        s1 = foo.replace('#include "common.cuh"\n', '#include "common.cuh"\n' + hook_inc).replace(
            "x[0] = 1.0f; }\n", "x[0] = 1.0f; }\nstatic int other_speed(int n) {\n    return n * 2;\n}\n").replace(
            "void foo_launch(float * x) { k_old<<<1, 1>>>(x); }\n", "void foo_launch(float * x) {\n" + hook_call + "}\n")
        branch(repo, "s1", {"ggml/src/ggml-cuda/foo.cu": s1})
        branch(repo, "s1b", {"ggml/src/ggml-cuda/foo.cu": s1.replace("pxqn_ext_gemv(x, 1)", "pxqn_ext_gemv(x, 2)")})

        allow = os.path.join(t, "allow.txt")
        with open(allow, "w") as f:
            f.write("# kind:value | reason | approver\n")
        out = os.path.join(t, "out.json")

        def case(name, args, want_rc, checks=(), allow_file=allow):
            nonlocal fails
            rc, js, txt = scan(repo, args, allow_file, out)
            bad = [c for c in checks if not has(js, *c)]
            ok = rc == want_rc and not bad
            print("%s  %-26s rc=%d want=%d hits=%d%s" % ("ok  " if ok else "FAIL", name, rc, want_rc, len(js.get("hits", [])),
                                                       "" if not bad else " missing=%s" % bad))
            if not ok:
                print("      " + txt.strip().replace("\n", "\n      ")[:1500])
            fails += not ok
            return js, txt

        js, txt = case("bad (kernel + getenv)", ["--ref", "bad"], 2,
                       [("new-kernel", "ggml/src/ggml-cuda/foo.cu", "k_dummy"),
                        ("new-lever", "src/llama.cpp", "PXA_FOO"),
                        ("speed-path-code", "ggml/src/ggml-cuda/foo.cu")])
        lines_ok = "ggml/src/ggml-cuda/foo.cu:7" in txt and "src/llama.cpp:9" in txt
        print("%s  %-26s file:line in output" % ("ok  " if lines_ok else "FAIL", "bad file:line"))
        fails += not lines_ok
        case("newcu (new .cu file)", ["--ref", "newcu"], 2, [("new-cuda-file", "ggml/src/ggml-cuda/bar.cu")])
        case("hook only", ["--ref", "hook"], 0)
        case("marker fallback", ["--ref", "marker"], 0)
        case("docs + comment-only", ["--ref", "docs"], 0)
        case("tests only", ["--ref", "tests"], 0)
        case("closed paths (ref)", ["--ref", "closed"], 0)
        case("report mode", ["--ref", "bad", "--report"], 0,
             [("new-kernel", "ggml/src/ggml-cuda/foo.cu", "k_dummy"), ("new-lever", "src/llama.cpp", "PXA_FOO")])

        allow2 = os.path.join(t, "allow2.txt")
        with open(allow2, "w") as f:
            f.write("# reviewed\nsym:PXA_FOO | debug switch, not a speed lever | reviewer-a\n"
                    "path:ggml/src/ggml-cuda/foo.cu | test fixture | reviewer-b\n")
        js, _ = case("allowlist silences", ["--ref", "bad"], 0, allow_file=allow2)
        allow3 = os.path.join(t, "allow3.txt")
        with open(allow3, "w") as f:
            f.write("sym:PXA_FOO | no approver\n")
        case("allowlist malformed", ["--ref", "bad"], 3, allow_file=allow3)
        allow4 = os.path.join(t, "allow4.txt")
        with open(allow4, "w") as f:
            f.write("sym:PXA_FOO | debug switch | reviewer-a\n")
        case("allowlist sym only", ["--ref", "bad"], 2, [("new-kernel", "ggml/src/ggml-cuda/foo.cu", "k_dummy")],
             allow_file=allow4)

        # hunk entries: the sha of a hunk = sha256 over its code lines (comment-only / blank lines dropped), each stripped,
        # joined by "\n"; computed here from the spec, and the report's own "hunk" field must agree
        def hsha(lines):
            return hashlib.sha256("\n".join(l.strip() for l in lines).encode()).hexdigest()
        h_inc = hsha([hook_inc.rstrip("\n")])
        h_call = hsha(["void foo_launch(float * x) {"] + hook_call.rstrip("\n").split("\n") + ["}"])
        js, _ = case("s1 no entry", ["--ref", "s1"], 2, [("speed-path-code", "ggml/src/ggml-cuda/foo.cu")])
        got = sorted(h.get("hunk") or "" for h in js.get("hits", []) if h.get("category") == "speed-path-code")
        n_s1 = len(got)
        sha_ok = h_inc in got and h_call in got and n_s1 == 3
        print("%s  %-26s %d speed hunks, spec sha in report=%s" % ("ok  " if sha_ok else "FAIL", "s1 hunk ids", n_s1,
                                                                   h_inc in got and h_call in got))
        fails += not sha_ok
        allow5 = os.path.join(t, "allow5.txt")
        with open(allow5, "w") as f:
            f.write("hunk:ggml/src/ggml-cuda/foo.cu:%s | S1 versioned hook include | reviewer-b\n" % h_inc)
            f.write("hunk:ggml/src/ggml-cuda/foo.cu:%s | S1 versioned hook call site | reviewer-b\n" % h_call)
        js, _ = case("s1 hunks allowed", ["--ref", "s1"], 2, [("speed-path-code", "ggml/src/ggml-cuda/foo.cu")],
                     allow_file=allow5)
        left = [h for h in js.get("hits", []) if h.get("category") == "speed-path-code"]
        one_ok = len(left) == 1 and left[0].get("hunk") not in (h_inc, h_call) and js.get("allowed") == 2
        print("%s  %-26s unrelated hunk still a hit (%d left, %s allowed)" % ("ok  " if one_ok else "FAIL",
                                                                            "s1 other hunk", len(left), js.get("allowed")))
        fails += not one_ok
        js, _ = case("s1b changed hunk", ["--ref", "s1b"], 2, allow_file=allow5)
        left = [h for h in js.get("hits", []) if h.get("category") == "speed-path-code"]
        ch_ok = len(left) == 2 and js.get("allowed") == 1
        print("%s  %-26s changed call site a hit again (%d hunks hit, %s allowed)" % (
            "ok  " if ch_ok else "FAIL", "s1b reopened", len(left), js.get("allowed")))
        fails += not ch_ok
        allow6 = os.path.join(t, "allow6.txt")
        with open(allow6, "w") as f:
            f.write("hunk:ggml/src/ggml-cuda/foo.cu:abc123 | short sha | reviewer-b\n")
        case("hunk entry malformed", ["--ref", "s1"], 3, allow_file=allow6)

        exp = os.path.join(t, "export-bad")
        git(repo, "worktree", "add", "-q", "--detach", os.path.join(t, "wt-bad"), "bad")
        shutil.copytree(os.path.join(t, "wt-bad"), exp, ignore=shutil.ignore_patterns(".git"))
        case("dir export of bad", [exp], 2, [("new-kernel", "ggml/src/ggml-cuda/foo.cu", "k_dummy"),
                                            ("new-lever", "src/llama.cpp", "PXA_FOO")])
        exp2 = os.path.join(t, "export-docs")
        git(repo, "worktree", "add", "-q", "--detach", os.path.join(t, "wt-docs"), "docs")
        shutil.copytree(os.path.join(t, "wt-docs"), exp2, ignore=shutil.ignore_patterns(".git", "pxa-closed-files.txt"))
        case("dir export of docs", [exp2, "--closed-list", os.path.join(repo, "scripts", "pxa-closed-files.txt")], 0)
        write(exp2, {"src/closed-dir/spec.cpp": "int f() { return 0; }\n"})
        case("dir with closed path", [exp2, "--closed-list", os.path.join(repo, "scripts", "pxa-closed-files.txt")], 2,
             [("closed-path-present", "src/closed-dir/spec.cpp")])
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("release-scan closed-lib: %s (%d failure(s))" % ("PASS" if not fails else "FAIL", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
