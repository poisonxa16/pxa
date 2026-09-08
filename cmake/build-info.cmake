set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")

# Look for git
find_package(Git)
if(NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if(GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
        message(STATUS "Found Git: ${GIT_EXECUTABLE}")
    else()
        message(WARNING "Git not found. Build info will not be accurate.")
    endif()
endif()

# Get the commit count and hash
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_COMMIT ${HEAD})
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_NUMBER ${COUNT})
    endif()
endif()

# 2026-09-06 (release packaging): a release tarball is built from a `git archive` extract,
# which by construction has no `.git`, so the discovery above cannot run and every packaged binary
# reported `version: 0 (unknown)` -- the one banner a user is most likely to quote in a bug report.
# The packaging script now passes the real values in, and an explicit value always wins over
# discovery. Both must be supplied together; supplying one alone is a configure-time error rather
# than a half-filled banner.
# Truthiness, not DEFINED: common/CMakeLists.txt forwards these to the generator process
# unconditionally, so on an ordinary git build they arrive defined-but-empty and must fall
# through to discovery rather than blanking the banner.
if(PXA_BUILD_NUMBER OR PXA_BUILD_COMMIT)
    if(NOT (PXA_BUILD_NUMBER AND PXA_BUILD_COMMIT))
        message(FATAL_ERROR "build-info: PXA_BUILD_NUMBER and PXA_BUILD_COMMIT must be given together")
    endif()
    set(BUILD_NUMBER ${PXA_BUILD_NUMBER})
    set(BUILD_COMMIT ${PXA_BUILD_COMMIT})
    message(STATUS "build-info: injected build number ${BUILD_NUMBER}, commit ${BUILD_COMMIT}")
endif()

if(MSVC)
    set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else()
    execute_process(
        COMMAND sh -c "$@ --version | head -1" _ ${CMAKE_C_COMPILER}
        OUTPUT_VARIABLE OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(BUILD_COMPILER ${OUT})
    execute_process(
        COMMAND ${CMAKE_C_COMPILER} -dumpmachine
        OUTPUT_VARIABLE OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(BUILD_TARGET ${OUT})
endif()
