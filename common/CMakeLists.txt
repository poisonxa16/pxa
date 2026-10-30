# common

find_package(Threads REQUIRED)

# Build info header
#

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.git")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else()
            set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${REAL_GIT_DIR}")
        endif()
    endif()

    if(EXISTS "${GIT_DIR}/index")
        set(GIT_INDEX "${GIT_DIR}/index")
    else()
        message(WARNING "Git index not found in git repository.")
        set(GIT_INDEX "")
    endif()
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
    set(GIT_INDEX "")
endif()

# Add a custom command to rebuild build-info.cpp when .git/index changes
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp"
    COMMENT "Generating build details from Git"
    COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
            -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DPXA_BUILD_NUMBER=${PXA_BUILD_NUMBER} -DPXA_BUILD_COMMIT=${PXA_BUILD_COMMIT}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info-gen-cpp.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in" ${GIT_INDEX}
    VERBATIM
)
set(TARGET build_info)
add_library(${TARGET} OBJECT build-info.cpp)
if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(TARGET common)

add_library(${TARGET} STATIC
    base64.hpp
    chat-auto-parser-generator.cpp
    chat-auto-parser-helpers.cpp
    chat-auto-parser.h
    chat-diff-analyzer.cpp
	chat-peg-parser.cpp
    chat-peg-parser.h
    common.cpp
    sampling.h
    sampling.cpp
    console.h
    console.cpp
    json-partial.h
    json-partial.cpp
    llguidance.cpp
    json-schema-to-grammar.cpp
    train.h
    train.cpp
    log.cpp
    log.h    
    http.h
    ngram-cache.cpp
    ngram-cache.h
    ngram-map.cpp
    ngram-map.h
    peg-parser.cpp
    peg-parser.h
    speculative.cpp
    pxa-registry.cpp
    pxa-registry.h
    spec-tuner.cpp
    spec-tuner.h
    unicode.cpp
    unicode.h
    ngram-mod.cpp
    ngram-mod.h
    suffix-tree.cpp
    suffix-tree.h
    regex-partial.cpp
    regex-partial.h
    reasoning-budget.cpp
    reasoning-budget.h
    chat.cpp
    chat.h
    jinja/lexer.cpp
    jinja/lexer.h
    jinja/parser.cpp
    jinja/parser.h
    jinja/runtime.cpp
    jinja/runtime.h
    jinja/value.cpp
    jinja/value.h
    jinja/string.cpp
    jinja/string.h
    jinja/caps.cpp
    jinja/caps.h
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(LLAMA_COMMON_EXTRA_LIBS build_info)

# Use curl to download model url
if (LLAMA_CURL)
    find_package(CURL REQUIRED)
    add_definitions(-DLLAMA_USE_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    find_library(CURL_LIBRARY curl REQUIRED)
    set(LLAMA_COMMON_EXTRA_LIBS ${LLAMA_COMMON_EXTRA_LIBS} ${CURL_LIBRARY})
endif ()

if (LLAMA_LLGUIDANCE)
    include(ExternalProject)
    set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source)
    set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)
    ExternalProject_Add(llguidance_ext
        GIT_REPOSITORY https://github.com/guidance-ai/llguidance
        # v0.6.12:
        GIT_TAG ced1c9023d47ec194fa977932d35ce65c2ebfc09
        PREFIX ${CMAKE_BINARY_DIR}/llguidance
        SOURCE_DIR ${LLGUIDANCE_SRC}
        BUILD_IN_SOURCE TRUE
        CONFIGURE_COMMAND ""
        # --package llguidance: only the C-API crate. A bare `cargo build` builds every default workspace member,
        # including toktrie_hf_tokenizers, whose openssl-sys build script fails without libssl-dev + pkg-config
        # (seen on a CPU build without them, gate-logs v31-llguidance). Same command as upstream llama.cpp.
        # --locked: the crate's own Cargo.lock at the pinned GIT_TAG is the exact dependency set; cargo refuses to
        # re-resolve it, so a build fetches the same crate versions every time.
        BUILD_COMMAND cargo build --release --locked --package llguidance
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${LLGUIDANCE_PATH}/libllguidance.a ${LLGUIDANCE_PATH}/llguidance.h
        UPDATE_COMMAND ""
    )
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_LLGUIDANCE)

    add_library(llguidance STATIC IMPORTED)
    set_target_properties(llguidance PROPERTIES IMPORTED_LOCATION ${LLGUIDANCE_PATH}/libllguidance.a)
    add_dependencies(llguidance llguidance_ext)

    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})
    set(LLAMA_COMMON_EXTRA_LIBS ${LLAMA_COMMON_EXTRA_LIBS} llguidance)
endif ()

# The registry compiles from a copy whose evidence strings (internal record ids) are empty; every other byte of the
# source is the same (scripts/pxa-release-scrub.py). Every build does this unless PXA_REGISTRY_EVIDENCE=ON (dev builds
# only, never a release: the package scan refuses the ids), so a dev build runs and logs exactly what a release ships.
option(PXA_REGISTRY_EVIDENCE "pxa: compile the registry's evidence ids into the binaries (dev builds only, never a release)" OFF)
if (PXA_REGISTRY_EVIDENCE AND PXA_RELEASE_BUILD)
    message(FATAL_ERROR "pxa: PXA_REGISTRY_EVIDENCE=ON is a dev build: it cannot be combined with PXA_RELEASE_BUILD=ON")
endif()
if (PXA_REGISTRY_EVIDENCE)
    message(WARNING "pxa: PXA_REGISTRY_EVIDENCE=ON compiles internal record ids into the binaries: a dev build, never a release")
    target_compile_definitions(${TARGET} PUBLIC PXA_REGISTRY_EVIDENCE_IDS=1)
else()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_pxa_reg_rel ${CMAKE_CURRENT_BINARY_DIR}/pxa-release/pxa-registry.cpp)
    add_custom_command(OUTPUT ${_pxa_reg_rel}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/pxa-release-scrub.py
                ${CMAKE_CURRENT_SOURCE_DIR}/pxa-registry.cpp ${_pxa_reg_rel}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pxa-registry.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/pxa-release-scrub.py
        VERBATIM)
    get_target_property(_pxa_common_srcs ${TARGET} SOURCES)
    list(REMOVE_ITEM _pxa_common_srcs pxa-registry.cpp)
    list(APPEND _pxa_common_srcs ${_pxa_reg_rel})
    set_target_properties(${TARGET} PROPERTIES SOURCES "${_pxa_common_srcs}")
endif()

target_include_directories(${TARGET} PUBLIC . ../vendor)
target_include_directories(${TARGET} PRIVATE ${PXA_CATALOG_DIR})   # pxa-lever-catalog-build.inc (top-level CMakeLists.txt)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)
target_link_libraries     (${TARGET} PRIVATE ${LLAMA_COMMON_EXTRA_LIBS} PUBLIC llama Threads::Threads)
