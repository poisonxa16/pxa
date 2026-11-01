// pxqn-plant-lib.c -- test-pxqn-loader-plant's stand-in for a planted PXQN library. Loading it is the failure:
// its load hook writes the file named by PXA_PLANT_MARKER (a DLL's DllMain runs before any ABI check could refuse it).
#include <stdio.h>
#include <stdlib.h>

static void pxa_plant_mark(void) {
    const char * m = getenv("PXA_PLANT_MARKER");
    if (!m || !*m) return;
    FILE * f = fopen(m, "wb");
    if (f) { fputs("planted library loaded\n", f); fclose(f); }
}

#if defined(_WIN32)
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
    (void) inst; (void) reserved;
    if (reason == DLL_PROCESS_ATTACH) pxa_plant_mark();
    return TRUE;
}
#else
__attribute__((constructor)) static void pxa_plant_ctor(void) { pxa_plant_mark(); }
#endif
