// test-pxqn-loader-plant -- the PXQN loader (ggml/src/ggml-pxqn-loader.cpp) never loads a library planted in the
// current folder, nor (Windows) in a %PATH% folder (4.0 security review M2: the bare-name LoadLibrary fallback).
// argv[1] = the stand-in library (pxqn-plant-lib.c); loading it writes the marker file named by PXA_PLANT_MARKER.
// The test copies it under the real library's file name into a fresh current folder (and a %PATH% folder on
// Windows), makes the engine load PXQN, and fails if the marker appears. Linux keeps its dlopen search (rpath,
// LD_LIBRARY_PATH, ld cache), which never includes the current folder, so there only the current folder is planted.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <process.h>
static const char * k_lib_name = "ggml-pxqn.dll";
static void set_env(const char * k, const std::string & v) { _putenv_s(k, v.c_str()); SetEnvironmentVariableA(k, v.c_str()); }
static int  pid(void) { return _getpid(); }
#else
#include <unistd.h>
static const char * k_lib_name = "libggml-pxqn.so";
static void set_env(const char * k, const std::string & v) { setenv(k, v.c_str(), 1); }
static int  pid(void) { return (int) getpid(); }
#endif

extern "C" bool ggml_pxqn_cpu_mmv_available(void);   // GGML_API in ggml-pxqn-api.h: first call loads the library

namespace fs = std::filesystem;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <planted-library>\n", argv[0]); return 2; }
    const fs::path plant = fs::absolute(argv[1]);
    std::error_code ec;
    if (!fs::exists(plant, ec)) { fprintf(stderr, "FAIL: no stand-in library at %s\n", plant.string().c_str()); return 1; }

    // pid + clock: not a guessable name another user could pre-create
    const fs::path root = fs::temp_directory_path() /
        ("pxqn-plant-" + std::to_string(pid()) + "-" +
         std::to_string((unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path cwd = root / "cwd", pathdir = root / "pathdir", marker = root / "marker";
    fs::remove_all(root, ec);
    fs::create_directories(cwd); fs::create_directories(pathdir);
    fs::copy_file(plant, cwd / k_lib_name, fs::copy_options::overwrite_existing);
#if defined(_WIN32)
    fs::copy_file(plant, pathdir / k_lib_name, fs::copy_options::overwrite_existing);
    const char * old_path = getenv("PATH");
    set_env("PATH", pathdir.string() + ";" + (old_path ? old_path : ""));
#endif
#if defined(_WIN32)
    _putenv_s("PXA_PXQN_LIB", ""); SetEnvironmentVariableA("PXA_PXQN_LIB", NULL);   // the explicit override stays the user's call
    _putenv_s("PXA_PXQN_DISABLE", ""); SetEnvironmentVariableA("PXA_PXQN_DISABLE", NULL);
#else
    unsetenv("PXA_PXQN_LIB"); unsetenv("PXA_PXQN_DISABLE");
    // glibc reads an empty LD_LIBRARY_PATH element as the current folder: drop those, so a caller's environment
    // cannot turn this into a false FAIL (the engine does not control the user's LD_LIBRARY_PATH)
    if (const char * llp = getenv("LD_LIBRARY_PATH")) {
        std::string in(llp), out;
        size_t i = 0;
        while (i <= in.size()) {
            size_t j = in.find(':', i);
            if (j == std::string::npos) j = in.size();
            if (j > i) out += (out.empty() ? "" : ":") + in.substr(i, j - i);
            i = j + 1;
        }
        set_env("LD_LIBRARY_PATH", out);
    }
#endif
    set_env("PXA_PXQN_VERBOSE", "1");
    set_env("PXA_PLANT_MARKER", marker.string());
    fs::current_path(cwd);

    const bool mmv = ggml_pxqn_cpu_mmv_available();
    fs::current_path(root.parent_path());
    const bool planted = fs::exists(marker, ec);
    fs::remove_all(root, ec);
    if (planted) {
        fprintf(stderr, "FAIL: the PXQN loader loaded the library planted in the current folder%s\n",
#if defined(_WIN32)
                " or on %PATH%"
#else
                ""
#endif
        );
        return 1;
    }
    // with a real library next to libggml the fallback never runs: say so, the Windows rig run (no real library)
    // is the one that exercises it
    printf("OK: no planted %s loaded (%s)\n", k_lib_name,
           mmv ? "a real library next to libggml loaded first: the fallback was not exercised"
               : "no real library: the fallback search ran and skipped the planted copies");
    return 0;
}
