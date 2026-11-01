// test-pxa-lan-nudge: llama-server warns when it listens beyond loopback with no API key (4.0.1 security, review
// L13); examples/server/pxa-lan-nudge.h, no server, no network.
#include "../examples/server/pxa-lan-nudge.h"

#include <cstdio>

static int g_fail = 0;

static void expect_loopback(const char * host, bool want) {
    const bool got = pxa_host_is_loopback(host);
    if (got != want) {
        fprintf(stderr, "FAIL: pxa_host_is_loopback(\"%s\") = %d, want %d\n", host, got, want);
        ++g_fail;
    }
}

static void expect_nudge(const char * host, size_t n_keys, bool want) {
    const bool got = !pxa_lan_nudge(host, n_keys).empty();
    if (got != want) {
        fprintf(stderr, "FAIL: pxa_lan_nudge(\"%s\", %zu) warns = %d, want %d\n", host, n_keys, got, want);
        ++g_fail;
    }
}

int main() {
    for (const char * h : { "127.0.0.1", "127.1.2.3", "127.255.255.255", "localhost", "::1", "[::1]",
                            "LOCALHOST", "localhost.", "::ffff:127.0.0.1", "[::ffff:127.0.0.1]", "127.0.0.1." }) {
        expect_loopback(h, true);
    }
    for (const char * h : { "0.0.0.0", "", "192.168.1.10", "10.0.0.1", "::", "[::]", "127.0.0.1.evil.example",
                            "127.0.0", "127.0.0.256", "127.0.0.1:8080", "1270.0.0.1", "localhost.example", "128.0.0.1",
                            "/tmp/llama.sock", "::ffff:192.168.1.10", "localhost..", "::ffff:" }) {
        expect_loopback(h, false);
    }
    expect_nudge("0.0.0.0", 0, true);
    expect_nudge("192.168.1.10", 0, true);
    expect_nudge("0.0.0.0", 1, false);
    expect_nudge("127.0.0.1", 0, false);
    expect_nudge("localhost", 0, false);
    // the advice names only options this server reads (it has no LLAMA_API_KEY variable)
    const std::string msg = pxa_lan_nudge("0.0.0.0", 0);
    if (msg.find("--api-key-file") == std::string::npos || msg.find("LLAMA_API_KEY") != std::string::npos) {
        fprintf(stderr, "FAIL: nudge text: %s\n", msg.c_str());
        ++g_fail;
    }
    if (g_fail) {
        fprintf(stderr, "test-pxa-lan-nudge: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-pxa-lan-nudge: OK\n");
    return 0;
}
