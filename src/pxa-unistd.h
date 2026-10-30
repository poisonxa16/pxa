// pxa-unistd.h -- <unistd.h> on POSIX, pxa-win-compat.h on Windows (MSVC). For files that included <unistd.h> directly.
#pragma once
#ifdef _WIN32
#include "pxa-win-compat.h"
#else
#include <unistd.h>
#endif
