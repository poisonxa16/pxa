// pxa-env-flag.h - one parse rule for PXA_* on/off environment switches (C and C++, header only).
//
// The lever contract is "=0 is the off switch". The idiom `!(e && *e && atoi(e) == 0)` read every
// NON-numeric value as 0, so PXA_X=on or PXA_X=true silently turned a default-on lever OFF (review of
// bug #332). The rule here:
//   unset, empty or blank             -> the default
//   a number (leading sign or digit)  -> exactly what atoi() gave before: 0 = off, anything else = on
//   off / false / no (any case)       -> off (these users already read off, and stay off)
//   anything else (on, true, yes ...) -> on
// Numeric values parse exactly as before, so every documented setting keeps its meaning; only the
// non-numeric words other than off/false/no change (from off to on).
//
// Usage:  static const bool on = pxa_env_flag("PXA_X", 1);          // default-on lever, =0 off
//         if (pxa_env_value_off(getenv("PXA_X"))) { ... }            // "is it explicitly off?"
// Lives in ggml/include so src/, common/, tools and ggml/src (C, C++ and CUDA host code) can all use it.
// Not part of the installed public ggml headers.
#pragma once

#include <ctype.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// case-insensitive "v (after leading blanks) is exactly word, then only blanks"
static inline int pxa_env_word_is_(const char * v, const char * word) {
    while (*word) {
        if (tolower((unsigned char) *v) != *word) { return 0; }
        ++v; ++word;
    }
    while (isspace((unsigned char) *v)) { ++v; }
    return *v == '\0';
}

// 1 when the value explicitly turns a switch off, 0 otherwise (NULL / empty / blank -> 0)
static inline int pxa_env_value_off(const char * v) {
    if (v == NULL) { return 0; }
    while (isspace((unsigned char) *v)) { ++v; }
    if (*v == '\0') { return 0; }
    if (isdigit((unsigned char) v[0]) || ((v[0] == '-' || v[0] == '+') && isdigit((unsigned char) v[1]))) {
        return atoi(v) == 0;
    }
    return pxa_env_word_is_(v, "off") || pxa_env_word_is_(v, "false") || pxa_env_word_is_(v, "no");
}

// the switch's value: def when unset / empty / blank, 0 when explicitly off, 1 otherwise
static inline int pxa_env_value_flag(const char * v, int def) {
    if (v == NULL) { return def; }
    const char * p = v;
    while (isspace((unsigned char) *p)) { ++p; }
    if (*p == '\0') { return def; }
    return !pxa_env_value_off(v);
}

static inline int pxa_env_flag(const char * name, int def) {
    return pxa_env_value_flag(getenv(name), def);
}

#ifdef __cplusplus
}
#endif
