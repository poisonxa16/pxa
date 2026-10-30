// Copyright (c) 2026 PXA Network. Part of PXA; distributed under the repository's licence (see LICENSE).
//
// test-pxa-stdout-guard: common/pxa-stdout-guard.h keeps a tool's data stdout clean (beta 2 report:
// `llama-bench -o json` had PXA status lines inside the JSON). The test runs itself as a child with
// stdout and stderr sent to two files, and checks:
//   1. guard on : the data stream holds ONLY the data; a stray printf/puts/fwrite(stdout) and a
//                 PXA-style status line land on stderr, nothing is lost
//   2. guard off (PXA_BENCH_SHARED_STDOUT=1): the old behaviour, everything on stdout
//   3. the data stream is a valid JSON document when the child writes one between the noise
//   4. the registry banner names an undeclared PXA_* variable but never prints its value (such a
//      variable can hold a token; these lines reach logs and bug reports)
#include "pxa-stdout-guard.h"
#include "pxa-registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static int child() {
    FILE * data = pxa_stdout_guard_take();
    std::printf("PXA_REGISTRY: stray status line via printf\n");
    std::fputs("PXA_AUTO: stray status line via fputs\n", stdout);
    std::fflush(stdout);
    std::fprintf(data, "[\n  {\"k\": 1}\n");
    std::fwrite("pxqn: stray fwrite\n", 1, 19, stdout);
    std::fflush(stdout);
    std::fprintf(data, "]\n");
    std::fprintf(stderr, "a real stderr line\n");
    pxa_stdout_guard_release(data);
    std::printf("PXA_TAIL: after release\n");
    return 0;
}

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int fails = 0;
static void check(bool ok, const char * what) {
    std::fprintf(stderr, "%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        fails++;
    }
}

static bool has(const std::string & s, const char * needle) { return s.find(needle) != std::string::npos; }

static void set_env(const char * k, const char * v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) { setenv(k, v, 1); } else { unsetenv(k); }
#endif
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "child") == 0) {
        return child();
    }
    const std::string self = argv[0];
    const std::string out = "test-pxa-stdout-guard.out", err = "test-pxa-stdout-guard.err";
    const std::string cmd = "\"" + self + "\" child > " + out + " 2> " + err;

    // 1 + 3: guard on
    set_env("PXA_BENCH_SHARED_STDOUT", nullptr);
    check(std::system(cmd.c_str()) == 0, "child ran (guard on)");
    std::string o = slurp(out), e = slurp(err);
#if defined(_WIN32)
    // the data stream is text mode on Windows, as stdout itself is: the same bytes an unguarded stdout gives there
    check(o == "[\r\n  {\"k\": 1}\r\n]\r\n", "guard on: stdout holds only the data, byte for byte");
#else
    check(o == "[\n  {\"k\": 1}\n]\n", "guard on: stdout holds only the data, byte for byte");
#endif
    check(has(e, "PXA_REGISTRY: stray status line via printf"), "guard on: printf line moved to stderr");
    check(has(e, "PXA_AUTO: stray status line via fputs"), "guard on: fputs line moved to stderr");
    check(has(e, "pxqn: stray fwrite"), "guard on: fwrite line moved to stderr");
    check(has(e, "a real stderr line"), "guard on: real stderr kept");
    check(has(e, "PXA_TAIL: after release"), "guard on: stdout after release still goes to stderr (no data corruption)");

    // 2: guard off
    set_env("PXA_BENCH_SHARED_STDOUT", "1");
    check(std::system(cmd.c_str()) == 0, "child ran (guard off)");
    o = slurp(out); e = slurp(err);
    check(has(o, "PXA_REGISTRY: stray status line via printf") && has(o, "{\"k\": 1}"), "guard off: old shared stdout");
    check(!has(e, "PXA_REGISTRY"), "guard off: nothing moved to stderr");
    set_env("PXA_BENCH_SHARED_STDOUT", "0");
    check(pxa_stdout_guard_disabled() == false, "PXA_BENCH_SHARED_STDOUT=0 means the guard stays on");
    set_env("PXA_BENCH_SHARED_STDOUT", nullptr);

    // 4: names only for an undeclared PXA_* variable
    set_env("PXA_TEST_NOT_A_LEVER_SECRET", "tok-0123456789abcdef");
    FILE * tf = std::tmpfile();
    check(tf != nullptr, "tmpfile");
    if (tf) {
        pxa_registry_banner(tf, pxa_topology{}, pxa_model_info{}, pxa_autoconfig{});
        std::fflush(tf);
        std::rewind(tf);
        std::string b;
        char buf[4096];
        size_t nr;
        while ((nr = std::fread(buf, 1, sizeof(buf), tf)) > 0) {
            b.append(buf, nr);
        }
        std::fclose(tf);
        check(has(b, "WARNING PXA_TEST_NOT_A_LEVER_SECRET is not a lever"), "banner names the undeclared variable");
        check(!has(b, "tok-0123456789abcdef"), "banner never prints the undeclared variable's value");
    }
    set_env("PXA_TEST_NOT_A_LEVER_SECRET", nullptr);

    std::fprintf(stderr, "%s (%d failure(s))\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
