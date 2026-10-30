// win-compat/strings.h -- Windows build only (include path of ggml/ggml-pxqn on WIN32): the POSIX <strings.h> calls used.
#pragma once
#include <string.h>
#define strcasecmp  _stricmp
#define strncasecmp _strnicmp
