#pragma once

// The LAN API-key nudge (4.0 security review L13): a server bound to anything but loopback with no --api-key
// serves the model to everyone on the network. llama-server warns at startup; it does not refuse (LAN serving
// without a key is a valid choice on a trusted network). Header-only so tests/test-pxa-lan-nudge.cpp runs
// without the server.

#include <cstddef>
#include <string>

// true for the loopback names and addresses: localhost (any case, optional trailing dot), 127.0.0.0/8 (also as
// ::ffff:127.x.y.z), ::1 (bracketed or not). Anything else warns: this server has no unix-socket bind, so every
// other name goes to the resolver and may reach the network.
static inline bool pxa_host_is_loopback(const std::string & host) {
    std::string h = host;
    if (h.size() >= 2 && h.front() == '[' && h.back() == ']') {
        h = h.substr(1, h.size() - 2);
    }
    for (char & c : h) {
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
    }
    if (!h.empty() && h.back() == '.') {
        h.pop_back();
    }
    if (h == "localhost" || h == "::1" || h == "0:0:0:0:0:0:0:1") {
        return true;
    }
    if (h.compare(0, 7, "::ffff:") == 0) {
        h = h.substr(7);
    }
    // 127.x.y.z: four dot-separated decimal parts, the first 127, each 0..255
    if (h.compare(0, 4, "127.") != 0) {
        return false;
    }
    int parts = 0;
    size_t i = 0;
    while (i <= h.size()) {
        size_t j = i;
        int v = 0;
        while (j < h.size() && h[j] >= '0' && h[j] <= '9' && j - i < 3) {
            v = v * 10 + (h[j] - '0');
            ++j;
        }
        if (j == i || v > 255) {
            return false;
        }
        ++parts;
        if (j == h.size()) {
            break;
        }
        if (h[j] != '.') {
            return false;
        }
        i = j + 1;
    }
    return parts == 4;
}

// the startup warning, or "" when none is due
static inline std::string pxa_lan_nudge(const std::string & host, size_t n_api_keys) {
    if (n_api_keys > 0 || pxa_host_is_loopback(host)) {
        return "";
    }
    return "listening on " + (host.empty() ? std::string("(all interfaces)") : host) +
           " with no API key: anyone who can reach this port can use the model. "
           "Set --api-key or --api-key-file, or bind --host 127.0.0.1.";
}
