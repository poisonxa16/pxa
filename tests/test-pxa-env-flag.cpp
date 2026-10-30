// pxa-env-flag.h: the one parse rule for PXA_* on/off switches (review of bug #332, low note 1).
// A non-numeric value used to read as atoi() == 0, so PXA_X=on / =true turned a default-on lever OFF.
// Checks: numbers parse exactly as atoi() did (0 off, any other number on), off/false/no (any case)
// are off, unset/empty/blank give the default, every other word is on. Pure host code, no model.
#include "pxa-env-flag.h"
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv / unsetenv
#endif

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;

static void expect_off(const char * v, int want) {
    const int got = pxa_env_value_off(v);
    if (got != want) {
        fprintf(stderr, "FAIL: pxa_env_value_off(%s%s%s) = %d, want %d\n", v ? "\"" : "", v ? v : "NULL", v ? "\"" : "", got, want);
        n_fail++;
    }
}

static void expect_flag(const char * v, int def, int want) {
    const int got = pxa_env_value_flag(v, def);
    if (got != want) {
        fprintf(stderr, "FAIL: pxa_env_value_flag(%s%s%s, %d) = %d, want %d\n", v ? "\"" : "", v ? v : "NULL", v ? "\"" : "", def, got, want);
        n_fail++;
    }
}

int main() {
    // explicit off
    const char * offs[] = { "0", "00", " 0", "\t0", "+0", "-0", "0 ", "0x1", "off", "OFF", "Off", "false", "FALSE", "False",
                            "no", "NO", "No", " off", "off ", "false\n" };
    for (const char * v : offs) { expect_off(v, 1); expect_flag(v, 1, 0); expect_flag(v, 0, 0); }

    // on: every other number, and every other word (the bug: on/true read as off)
    const char * ons[] = { "1", "2", "-1", "+1", "10", " 1", "on", "ON", "On", "true", "TRUE", "True", "yes", "YES",
                           "y", "enable", "1x", "offf", "nope", "false1", "of", "o", "-", "+" };
    for (const char * v : ons) { expect_off(v, 0); expect_flag(v, 1, 1); expect_flag(v, 0, 1); }

    // unset / empty / blank: the default
    const char * defs[] = { nullptr, "", " ", "\t", "  \n" };
    for (const char * v : defs) { expect_off(v, 0); expect_flag(v, 1, 1); expect_flag(v, 0, 0); }

    // numbers parse exactly as the old atoi() idiom did
    const char * nums[] = { "0", "1", "2", "-1", "00", "007", "-0", "+3", " 0", " 5", "0x10", "12abc", "0abc" };
    for (const char * v : nums) {
        const int old_off = atoi(v) == 0;   // the old `e && *e && atoi(e) == 0`
        expect_off(v, old_off);
    }

    // through the environment
    const char * name = "PXA_TEST_ENV_FLAG_332";
    unsetenv(name);
    if (pxa_env_flag(name, 1) != 1 || pxa_env_flag(name, 0) != 0) { fprintf(stderr, "FAIL: unset != default\n"); n_fail++; }
    setenv(name, "", 1);
    if (pxa_env_flag(name, 1) != 1 || pxa_env_flag(name, 0) != 0) { fprintf(stderr, "FAIL: empty != default\n"); n_fail++; }
    setenv(name, "true", 1);
    if (pxa_env_flag(name, 1) != 1 || pxa_env_flag(name, 0) != 1) { fprintf(stderr, "FAIL: true != on\n"); n_fail++; }
    setenv(name, "on", 1);
    if (pxa_env_flag(name, 1) != 1 || pxa_env_flag(name, 0) != 1) { fprintf(stderr, "FAIL: on != on\n"); n_fail++; }
    setenv(name, "0", 1);
    if (pxa_env_flag(name, 1) != 0 || pxa_env_flag(name, 0) != 0) { fprintf(stderr, "FAIL: 0 != off\n"); n_fail++; }
    setenv(name, "Off", 1);
    if (pxa_env_flag(name, 1) != 0) { fprintf(stderr, "FAIL: Off != off\n"); n_fail++; }
    unsetenv(name);

    if (n_fail) {
        fprintf(stderr, "test-pxa-env-flag: %d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-pxa-env-flag: OK (0/off/false/no off, unset/empty default, everything else on; numbers as atoi)\n");
    return 0;
}
