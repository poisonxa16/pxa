cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("pxa" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    include(git-vars)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# force MSVC compiler charset to utf-8
if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# extra artifacts
option(LLAMA_BUILD_TESTS    "llama: build tests"          ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES "llama: build examples"       ${LLAMA_STANDALONE})
option(LLAMA_BUILD_SERVER   "llama: build server example" ${LLAMA_STANDALONE})

# PXQ encoder. ON builds the code that WRITES PXQ files (the tier codecs, the backbone
# allocation table and the tier policy, all under src/pxq-encoder/) and the `pxq-quantize`
# tool that exposes it. OFF builds a tree that reads and runs PXQ files exactly as before but
# cannot write one: llama-quantize refuses a PXQ target and every other type is unaffected.
option(PXA_PXQ_ENCODER "pxa: build the PXQ encoder and the pxq-quantize tool" ON)
set(PXA_PXQ_QUANTIZER_URL "<<QUANTIZER-URL>>" CACHE STRING
    "pxa: where a build without the encoder tells the user to get pxq-quantize")

# PXQN (PXQ-Next) encoder: the private half of the quantizer (src/pxq-encoder/pxqn-*). ON is our own
# build; OFF is the PUBLIC pxq-quantize, which writes the classic PXQ tiers only. The public build
# is made from a tree that does not contain these files at all (release/v2026.10/quantizer/), so
# OFF is not a switch over code that is still in the binary: the code is not compiled.
# Copyright (c) 2026 PXA Network.
option(PXA_PXQN_ENCODER "pxa: build the private PXQN encoder into the quantizer" ${PXA_PXQ_ENCODER})
# PXQN kernels (owner decision 2026-09-28): the PXQN decode (CUDA kernels, the CPU reference
# decoder, the PXQN tail of the tensor-split epilogue and the PXQ4 row-block GEMV) is CLOSED. ON = this private tree
# builds it from source into libggml-pxqn.so (next to libggml.so, loaded at run time); the public tree has no such
# sources and builds only the open forwards, which run classic PXQ / k-quants and refuse PXQN files at load.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/pxqn-cpu.c)
    set(_pxa_pxqn_closed_default ON)
else()
    set(_pxa_pxqn_closed_default OFF)
endif()
option(PXA_PXQN_CLOSED_SRC "pxa: build the closed PXQN library (libggml-pxqn) from source" ${_pxa_pxqn_closed_default})
if (PXA_PXQN_CLOSED_SRC AND NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/pxqn-cpu.c)
    message(FATAL_ERROR "PXA_PXQN_CLOSED_SRC=ON but this tree has no closed PXQN sources (the public tree): "
                        "configure with -DPXA_PXQN_CLOSED_SRC=OFF")
endif()
# libggml-pxqn host code for baseline x86-64 (SSE2) so one library loads on any x86-64 CPU; the CPU-hot loops get an AVX2
# clone chosen at run time (ggml/src/CMakeLists.txt, pxqn-cpu.c). OFF = inherit the engine's -mavx/-mavx2/-mfma/-mf16c.
option(PXA_PXQN_BASELINE_X86 "pxa: build libggml-pxqn host code for baseline x86-64 with run-time AVX2 dispatch" ON)
set(PXA_QUANTIZER_VERSION "dev" CACHE STRING "pxa: the release tag the quantizer reports and writes")
# Which PXQN quantizer this is (docs/QUANTIZER-TIERS-AND-SERVICE-2026-09-26.md), only with PXA_PXQN_ENCODER=ON:
#   owner  PXA's own build (the supporter quant service): every PXQN path incl. LDLQ, no license check.
#   pro    the distributed Pro build: PXQN tiers, allocator, tier maps and measure mode with RTN rounding,
#          each run needs a signed Pro license token; the LDLQ code is not compiled in at all.
# PXA_LICENSE_PUBKEYS ("kid:hex32;kid2:hex32", printed by the licence tool's pubkey command) is the set of
# Ed25519 public keys a pro build trusts; it is compiled in, never read from a path.
set(PXA_QUANTIZER_EDITION "owner" CACHE STRING "pxa: PXQN quantizer edition, owner or pro")
set(PXA_LICENSE_PUBKEYS "" CACHE STRING "pxa: license public keys a pro quantizer trusts (kid:hex;...)")
# The licence server's lock-signing public key(s) libggml-pxqn trusts to open locked (Pro, supporter / personal) models: "kid:hex64;kid2:hex64". Compiled in; never read from a path.
set(PXA_LOCK_PUBKEYS "" CACHE STRING "pxa: lock-signing public keys libggml-pxqn trusts for locked models (kid:hex;...)")
if (PXA_PXQN_ENCODER)
    if (NOT PXA_PXQ_ENCODER)
        message(FATAL_ERROR "PXA_PXQN_ENCODER=ON needs PXA_PXQ_ENCODER=ON")
    endif()
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/pxq-encoder/pxqn-quantize.inc.cpp)
        message(FATAL_ERROR "PXA_PXQN_ENCODER=ON but this tree has no src/pxq-encoder/pxqn-quantize.inc.cpp "
                            "(the public quantizer tree): configure with -DPXA_PXQN_ENCODER=OFF")
    endif()
    if (NOT PXA_QUANTIZER_EDITION STREQUAL "owner" AND NOT PXA_QUANTIZER_EDITION STREQUAL "pro")
        message(FATAL_ERROR "PXA_QUANTIZER_EDITION must be owner or pro (got '${PXA_QUANTIZER_EDITION}')")
    endif()
    if (PXA_QUANTIZER_EDITION STREQUAL "pro")
        # each entry kid:<64 lowercase hex> (CMake regex has no {n}: check the hex length apart)
        set(_pxa_pk_ok FALSE)
        foreach(_pk IN LISTS PXA_LICENSE_PUBKEYS)
            set(_pxa_pk_ok TRUE)
            if (NOT _pk MATCHES "^[A-Za-z0-9._-]+:([0-9a-f]+)$")
                set(_pxa_pk_ok FALSE)
                break()
            endif()
            string(LENGTH "${CMAKE_MATCH_1}" _pk_len)
            if (NOT _pk_len EQUAL 64)
                set(_pxa_pk_ok FALSE)
                break()
            endif()
        endforeach()
        if (NOT _pxa_pk_ok)
            message(FATAL_ERROR "PXA_QUANTIZER_EDITION=pro needs PXA_LICENSE_PUBKEYS=kid:<64 hex>[;...] "
                                "(printed by the licence tool's pubkey command)")
        endif()
    endif()
endif()

# 3rd party libs
option(LLAMA_CURL "llama: use libcurl to download model from an URL" OFF)
option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

# override ggml options
set(GGML_SANITIZE_THREAD    ${LLAMA_SANITIZE_THREAD})
set(GGML_SANITIZE_ADDRESS   ${LLAMA_SANITIZE_ADDRESS})
set(GGML_SANITIZE_UNDEFINED ${LLAMA_SANITIZE_UNDEFINED})
set(GGML_ALL_WARNINGS       ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE ON)
endif()

if (NOT DEFINED GGML_CUDA_USE_GRAPHS)
    set(GGML_CUDA_USE_GRAPHS ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

llama_option_depr(FATAL_ERROR LLAMA_CUBLAS              GGML_CUDA)
llama_option_depr(WARNING     LLAMA_CUDA                GGML_CUDA)
llama_option_depr(WARNING     LLAMA_KOMPUTE             GGML_KOMPUTE)
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL)
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC)
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL)
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16)
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN)

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        message(STATUS "Using -fsanitize=thread")

        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        message(STATUS "Using -fsanitize=address")

        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        message(STATUS "Using -fsanitize=undefined")

        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

#
# build the library
#

if (NOT TARGET ggml)
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# build the library
#

add_subdirectory(src)

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")


# At the moment some compile definitions are placed within the ggml/src
# directory but not exported on the `ggml` target. This could be improved by
# determining _precisely_ which defines are necessary for the llama-config
# package.
#
get_target_property(GGML_DIRECTORY ggml SOURCE_DIR)
get_directory_property(GGML_DIR_DEFINES DIRECTORY ${GGML_DIRECTORY} COMPILE_DEFINITIONS)
get_target_property(GGML_TARGET_DEFINES ggml COMPILE_DEFINITIONS)
set(GGML_TRANSIENT_DEFINES ${GGML_TARGET_DEFINES} ${GGML_DIR_DEFINES})
get_target_property(GGML_LINK_LIBRARIES ggml LINK_LIBRARIES)

set_target_properties(llama PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h)
install(TARGETS llama LIBRARY PUBLIC_HEADER)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
              LLAMA_LIB_INSTALL_DIR
              LLAMA_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake
    VERSION ${LLAMA_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama)

install(
    FILES convert_hf_to_gguf.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(cmake/llama.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        DESTINATION lib/pkgconfig)

#
# programs, examples and tests
#

add_subdirectory(common)
add_subdirectory(vendor/cpp-httplib)
if (LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()

# PXA binary names (2026-09-25). pxa-server, pxa-bench, pxa-quantize, pxa-perplexity and pxa-cli
# are created next to the llama-* binaries as symlinks, so both names run the same file and every
# existing script, recipe and docker line keeps working. Called from each tool's own CMakeLists
# (a POST_BUILD step has to live in the directory that creates the target).
# Copyright (c) 2026 PXA Network.
function(pxa_add_alias target alias)
    if (NOT WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:${target}> $<TARGET_FILE_DIR:${target}>/${alias}
            VERBATIM)
        install(CODE "execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink ${target} \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/${alias})")
    endif()
endfunction()

if (LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()

