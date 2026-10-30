// bug #324: the startup "lab knobs set" warning must name only lab knobs the USER put in the process environment.
// This helper runs llama_backend_init() after doing what the engine does to its own environment during startup
// (the registry's setenv picks, the tensor-split reduce default, the auto-spec switches ...) and prints the
// warning lines that llama_backend_init logged, verbatim, one per line. The driver (test-pxa-lab-knobs.sh) runs it
// under `env -i` with different environments and compares.
//
//   test-pxa-lab-knobs [engine-sets] [anything else = ignored, stands for CLI flags]
#include "llama.h"
#ifdef _WIN32
#include "pxa-win-compat.h"   // setenv
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void on_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN && strstr(text, "lab knobs") != nullptr) {
        fputs(text, stdout);
    }
}

int main(int argc, char ** argv) {
    llama_log_set(on_log, nullptr);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "engine-sets") == 0) {
            // what common/pxa-registry.cpp, the server's auto-spec choice and the launcher-less defaults do
            setenv("PXA_TSPLIT_REDUCE", "fused", 1);
            setenv("PXA_AUTO_SPEC_PICK", "1", 0);
            setenv("PXA_SPEC_AUTO_ARMED", "1", 1);
            setenv("PXA_LOGITS_CAP", "1", 0);
            setenv("PXA_XCACHE_VERIFY_WIDTH", "4", 0);
            setenv("PXQN_REGISTRY_DEFAULT", "1", 1);
        }
    }
    llama_backend_init();
    llama_backend_free();
    return 0;
}
