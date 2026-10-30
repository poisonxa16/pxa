#llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_target_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
add_executable(test-tokenizer-0 test-tokenizer-0.cpp)
target_link_libraries(test-tokenizer-0 PRIVATE common)
install(TARGETS test-tokenizer-0 RUNTIME)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
# server ignore_eos bans every EOG token (#361/#362)
llama_build_and_test(test-eog-tokens.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
llama_build_and_test(test-reasoning-budget-fit.cpp)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

# HF tokenizer.json parity incl. the server chat path (Metaspace prepend_scheme "first", BOS = chat start token)
add_executable(test-tokenizer-hf-parity test-tokenizer-hf-parity.cpp)
target_link_libraries(test-tokenizer-hf-parity PRIVATE common)
llama_test(test-tokenizer-hf-parity NAME test-tokenizer-hf-parity-nanbeige ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-nanbeige.gguf)
# the chat BOS-keep probe changes nothing outside prepend_scheme "first" vocabs (every repo vocab + a crafted RWKV one)
file(GLOB PXA_CHAT_BOS_VOCABS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-*.gguf)
list(SORT PXA_CHAT_BOS_VOCABS)
list(PREPEND PXA_CHAT_BOS_VOCABS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
add_executable(test-chat-bos-probe test-chat-bos-probe.cpp)
target_link_libraries(test-chat-bos-probe PRIVATE common)
llama_test(test-chat-bos-probe NAME test-chat-bos-probe ARGS ${PXA_CHAT_BOS_VOCABS})
llama_test(test-chat-bos-probe NAME test-chat-bos-probe-off ARGS ${PXA_CHAT_BOS_VOCABS})
set_tests_properties(test-chat-bos-probe-off PROPERTIES ENVIRONMENT "PXA_TOKENIZER_PREPEND_FIRST=0")

if (LLAMA_LLGUIDANCE)
    llama_target_and_test(test-grammar-llguidance.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
endif ()

# The release image (docker/Dockerfile) keeps building LLGuidance with a pinned Rust in the build stage only: a static
# check of the Dockerfiles and the llguidance cmake recipe (no docker, no network). The build + docker modes of the same
# script are run by hand: tests/test-llguidance-image.sh --build | --docker.
find_program(PXA_BASH bash)
if (PXA_BASH AND NOT WIN32)
    add_test(NAME test-llguidance-image COMMAND ${PXA_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/test-llguidance-image.sh --static)
    set_tests_properties(test-llguidance-image PROPERTIES SKIP_RETURN_CODE 77 LABELS "main")
endif ()

# The release tarball's asset name: "-sm<list>" derived from the CUDA arch list the package carries, plus the legacy
# "-sm60_61_70" alias (same bytes, own .sha256) over the 60;61;70 floor; an explicit ARCH_TAG as given. No docker, no build.
if (PXA_BASH AND NOT WIN32)
    add_test(NAME test-release-tarball-name COMMAND ${PXA_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/test-release-tarball-name.sh)
    set_tests_properties(test-release-tarball-name PROPERTIES LABELS "main")
endif ()

if (NOT WIN32)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API
    #llama_target_and_test(test-sampling.cpp)
    #llama_target_and_test(test-grammar-parser.cpp)
    #llama_target_and_test(test-grammar-integration.cpp)
    #llama_target_and_test(test-llama-grammar.cpp)
    #llama_target_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    #if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    #    llama_target_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    #    target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
    #endif()


    # build test-tokenizer-1-bpe target once and add many tests
    # add_executable(test-tokenizer-1-bpe test-tokenizer-1-bpe.cpp)
    # target_link_libraries(test-tokenizer-1-bpe PRIVATE common)
    # install(TARGETS test-tokenizer-1-bpe RUNTIME)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    # add_executable(test-tokenizer-1-spm test-tokenizer-1-spm.cpp)
    # target_link_libraries(test-tokenizer-1-spm PRIVATE common)
    # install(TARGETS test-tokenizer-1-spm RUNTIME)

    # llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

    # llama_target_and_test(test-double-float.cpp) # SLOW
endif()

#llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
# without python3 + jinja2 there is nothing to cross-check against: the test prints SKIPPED and exits 77
set_tests_properties(test-jinja-py PROPERTIES SKIP_RETURN_CODE 77)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp )
# a failed chat template leaves a null templates object: every accessor is safe on it
llama_build_and_test(test-chat-null-tmpls.cpp)
# chat-template-system-msg: late system messages merged to the front for strict templates only
llama_build_and_test(test-chat-system-merge.cpp ARGS ${PROJECT_SOURCE_DIR}/models/templates)
# glm-think-off: enable_thinking=false closes GLM-5.3's forced-open think block; every other template renders as before
llama_build_and_test(test-chat-glm-think.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
# bug #216: server_slot::update_chat_msg's parse decision and fallbacks. Header-only
# (examples/server/server-chat-parse.h), so no server and no model.
llama_build_and_test(test-server-chat-parse.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
target_include_directories(test-server-chat-parse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
llama_build_and_test(test-json-partial.cpp )
#llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

# llama_target_and_test(test-opt.cpp) # SLOW

llama_target_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_target_and_test(test-autorelease.cpp        LABEL "model")

# DeepSeek-V4 CUDA correctness harnesses. Both need a real CUDA device (a few hundred MB
# of VRAM each), so they are LABEL "cuda" and excluded from the default ctest run:
#     ctest -L cuda
# They are registered here on purpose. They ARE the correctness basis for the DSA and
# flipped-RoPE ports, and a correctness basis that only exists as an ad-hoc compile line
# on one borrowed machine is a correctness basis that will be lost.
#
# test-dsa-attn is deliberately registered WITHOUT PXA_DSA_ATTN=1: in that mode it
# asserts the shipped default (DSA declined). Run it with PXA_DSA_ATTN=1 by hand to
# exercise the kernel numerics.
# PXA_KV_SEQ_SOA parity harnesses (CPU only, no model for the first; the second takes a tiny gguf).
# CUDA op conformance/perf harness (mode test|perf, -o op); PXA_BO_MOE_SHAPES=1 adds the Gemma 4 routed-expert shapes
llama_build(test-backend-ops.cpp)

# PXA_P60_MEGA identity: the default one-token PXQN4 MoE half of the P60 stack against the
# reference reimplementation of the normal decode path on fixed random inputs, bitwise (ggml-cuda-p60.h hook through
# the PXQN CUDA table). Needs a card: LABEL cuda (ctest -L cuda); a CPU build SKIPs.
llama_build(test-pxa-p60-halves.cpp)
target_link_libraries(test-pxa-p60-halves PRIVATE ggml)
llama_test(test-pxa-p60-halves LABEL "cuda")
# PXA_FA_VEC_NY1_PAD: a verify batch straddling a 256 KV pad is bitwise its per-column decode
# (PXA_FA_VEC_NY1_SPLIT=1, sm_60 f32 vec route). Needs a card: LABEL cuda; a CPU build SKIPs.
llama_build(test-pxa-fa-ny1-pad.cpp)
target_link_libraries(test-pxa-fa-ny1-pad PRIVATE ggml)
llama_test(test-pxa-fa-ny1-pad LABEL "cuda")
set_tests_properties(test-pxa-fa-ny1-pad PROPERTIES SKIP_RETURN_CODE 77)
# FA stream-k with KV bounds (SWA KV_min/KV_max, KV_max at ne01>=1024): a slice clamped empty must not count a foreign
# tile twice (fattn-mma-f16 / fattn-new-mma, sm_75+). Device output vs a double-precision host reference. LABEL cuda.
llama_build(test-fa-mma-empty-slice.cpp)
target_link_libraries(test-fa-mma-empty-slice PRIVATE ggml)
llama_test(test-fa-mma-empty-slice LABEL "cuda")

# pass 8e review fixes: PXA_HOST_RESIDENT gate (R6/R7) and the epilogue cached-plan stash policy (R8), host-only
llama_build(test-pxa-host-resident.cpp)
llama_test(test-pxa-host-resident)
llama_build(test-pxa-epi-stash.cpp)
llama_test(test-pxa-epi-stash)

llama_build(test-kv-seq-shadow.cpp)
target_include_directories(test-kv-seq-shadow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-kv-seq-shadow)
llama_build(test-kv-mask-parity.cpp)
target_link_libraries(test-kv-mask-parity PRIVATE ggml)
# KV BAND ALIGNMENT: a sequence's logits must not depend on the cell its band starts at. Builds its
# own tiny dense model, so it is hermetic. CPU only, no model file, no card.
# PXQN activation dump: fp32 -> fp16 saturates beyond +-65504 (never inf), bit-identical within range. CPU only.
llama_build(test-pxqn-dump-sat.cpp)
target_link_libraries(test-pxqn-dump-sat PRIVATE ggml)
llama_test(test-pxqn-dump-sat)
llama_build(test-gallocr-leaf-reuse.cpp)
target_link_libraries(test-gallocr-leaf-reuse PRIVATE ggml)
llama_test(test-gallocr-leaf-reuse)
# bug #348: speculative verify graph slots are sized by a measure-only compute-buffer plan; it must equal a real
# reserve, allocate nothing, leave the allocator usable, and grow with n_kv. CPU only, no model file, no card.
llama_build(test-vslot-measure.cpp)
target_link_libraries(test-vslot-measure PRIVATE ggml)
llama_test(test-vslot-measure)
# CPU buffer bounds: the reported size must be usable from the aligned base (glm5next MTP n_max=4 heap
# corruption: a re-plan grown by one 32-byte unit kept a buffer 16 bytes too small). CPU only.
llama_build(test-cpu-buffer-bounds.cpp)
target_link_libraries(test-cpu-buffer-bounds PRIVATE ggml)
target_include_directories(test-cpu-buffer-bounds PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
llama_test(test-cpu-buffer-bounds)
llama_build(test-moe-merged-host-split.cpp)
target_link_libraries(test-moe-merged-host-split PRIVATE ggml)
llama_test(test-moe-merged-host-split)
llama_build(test-kv-band-align.cpp)
target_link_libraries(test-kv-band-align PRIVATE ggml)
llama_test(test-kv-band-align)
# the same property with every layer and the cache on a card (prints "nothing to do" and passes without one)
llama_test(test-kv-band-align NAME test-kv-band-align-gpu ARGS gpu)
# the non-GQA model (KV heads = query heads): the strided P.V route, host-side and on a card
llama_test(test-kv-band-align NAME test-kv-band-align-mha ARGS mha)
llama_test(test-kv-band-align NAME test-kv-band-align-gpu-mha ARGS gpu-mha)
# WARMUP GRAPH REUSE: graph reuse must never change a greedy answer. The load-time warmup (one BOS at
# n_eval 0) builds the all-experts MoE graph; reusing it for later 1-token decodes routed them through
# every expert (Gemma 4 cross-request carry, 10-17). Builds its own tiny MoE, so it is hermetic. CPU only.
llama_build(test-warmup-graph-reuse.cpp)
target_link_libraries(test-warmup-graph-reuse PRIVATE ggml)
llama_test(test-warmup-graph-reuse)
# GEMMA 4 -FA OFF V STORE (bug #332): -fa off stores V transposed, and a per-head Gemma 4 V
# [head_dim, n_head_kv, n_tokens] was transposed on the wrong axes, scrambling the whole V cache
# (-fa off / PXA_MODE=max answered garbage). Builds its own tiny dense Gemma 4 (two sliding blocks,
# one one-KV-head global block with V = K) and checks -fa off against -fa on. CPU only.
llama_build(test-gemma4-fa-off-vstore.cpp)
target_link_libraries(test-gemma4-fa-off-vstore PRIVATE ggml)
llama_test(test-gemma4-fa-off-vstore)
# The negative arm: the same test with the fix switched off (PXA_VTRANS_2D=0, the documented off switch)
# must FAIL, which proves the test catches the scrambled store. CPU only like the positive arm, and
# deterministic: the -fa off arms move the logits far past the tolerance on the old store. A setup
# failure (no model, no decode) is reported as SKIPPED here, never as a pass; the positive arm above
# fails on it anyway.
llama_test(test-gemma4-fa-off-vstore NAME test-gemma4-fa-off-vstore-bug)
set_tests_properties(test-gemma4-fa-off-vstore-bug PROPERTIES
    ENVIRONMENT "PXA_VTRANS_2D=0"
    WILL_FAIL TRUE
    SKIP_REGULAR_EXPRESSION "did not decode;could not write the test model;load failed")
# Both arms write and remove the same ./test-gemma4-fa-off-vstore.gguf in the test dir: never run them at once
# (ctest -j raced them: one arm removed the model while the other loaded it -> "load failed").
set_tests_properties(test-gemma4-fa-off-vstore test-gemma4-fa-off-vstore-bug PROPERTIES RESOURCE_LOCK gemma4-fa-off-vstore-gguf)
# pxa-env-flag.h: one parse rule for PXA_* on/off switches (0/off/false/no off, unset/empty default,
# anything else on; numbers exactly as atoi). Bug #332 review: PXA_X=on/true used to read as off.
llama_build(test-pxa-env-flag.cpp)
llama_test(test-pxa-env-flag)
# The same property one level down: the tile flash-attention kernel's running state must be fp32,
# or the 64-cell tile grouping (which is a function of the CELL index) reassociates a 20k-term
# accumulation differently at every band offset. Mirrors the kernel's arithmetic. CPU only.
llama_build(test-fa-band-align.cpp)
llama_test(test-fa-band-align)
# Bug #22738: every (ncols1, ncols2) the sm_70 vendored-MMA flash-attention dispatcher can launch has
# device code on the arch it runs on (sm_60 / sm_61 / sm_70), read from the same shape table the kernel
# guards and the dispatcher use (pxa/volta-mma/fattn-mma-shapes.h). CPU only.
llama_build(test-fa-volta-mma-shapes.cpp)
llama_test(test-fa-volta-mma-shapes)

llama_build(test-fa-tile-v2.cpp)
llama_test(test-fa-tile-v2)
# PXA_KV_SEQ_RM_INDEX differential harness: the indexed seq_rm against a verbatim copy of the old
# whole-cache scan, over randomized cache mutation traces. CPU only, no model.
llama_build(test-kv-seq-rm-index.cpp)
target_include_directories(test-kv-seq-rm-index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-kv-seq-rm-index)
# PXA_KV_CELL_MAX_EXACT: the attention window must cover every live cell of a unified ring, in
# particular a band that floats after a forced full re-processing. Keeps a verbatim copy of the
# retired sampled scan so the defect stays visible. CPU only, no model.
llama_build(test-kv-cell-max.cpp)
target_include_directories(test-kv-cell-max PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-kv-cell-max)
# PXA_PLE_DIRECT: the O_DIRECT PLE row reader (header-only, src/llama-pxa-ple-direct.h) against a file
# mapping on synthetic tables (Q8_0 / Q5_0 / F32 row sizes, rows straddling 4 KiB blocks, every cache
# size). CPU only, no model; SKIPs when the build tree's filesystem refuses O_DIRECT. `--gguf FILE`
# runs the same comparison on a real table, `--bench N` measures read latency.
llama_build(test-pxa-ple-direct.cpp)
target_include_directories(test-pxa-ple-direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-ple-direct)
# Hybrid KV / recurrent-state regressions from the 2026-09-25 bug hunt (#210 #214 #218 #223 #231
# #267): partial restore at np2, defrag vs the state-row permutation and the V layout, no replayed
# ubatches, gpu-fallback at np2, and the qwen4exp PLE window across a speculative rollback. CPU
# only. The ctest driver generates the fixtures with tests/gen_tiny_qwen4exp.py (plain, and --ple)
# (and --ple --state 64, bug #342) into the build tree and runs
#     test-kv-hybrid-state tiny-qwen4exp.gguf tiny-qwen4exp-ple.gguf tiny-qwen4exp-ple-s64.gguf
# It is labelled "model" and is reported skipped where python3 or numpy is missing.
llama_build(test-kv-hybrid-state.cpp)
find_package(Python3 COMPONENTS Interpreter QUIET)
add_test(NAME test-kv-hybrid-state
    COMMAND ${CMAKE_COMMAND}
        -DPYTHON=${Python3_EXECUTABLE}
        -DGEN=${CMAKE_CURRENT_SOURCE_DIR}/gen_tiny_qwen4exp.py
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/kv-hybrid-state-fixtures
        -DEXE=$<TARGET_FILE:test-kv-hybrid-state>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run-kv-hybrid-state.cmake)
set_tests_properties(test-kv-hybrid-state PROPERTIES LABELS "model" SKIP_REGULAR_EXPRESSION "KV_HYBRID_SKIP")
# bug #342: the same run with every layer offloaded (-ngl 99), so the recurrent states live on the card. The 32-wide
# fixture is a head width the CUDA delta-net kernel does not take: per-step checkpoints must step aside there (they
# read back stale rows before the fix); the 64-wide fixture must keep per-step. Without a card it repeats the CPU run.
if (GGML_CUDA)
    add_test(NAME test-kv-hybrid-state-offload
        COMMAND ${CMAKE_COMMAND}
            -DPYTHON=${Python3_EXECUTABLE}
            -DGEN=${CMAKE_CURRENT_SOURCE_DIR}/gen_tiny_qwen4exp.py
            -DOUT=${CMAKE_CURRENT_BINARY_DIR}/kv-hybrid-state-fixtures-offload
            -DEXE=$<TARGET_FILE:test-kv-hybrid-state>
            -DNGL=99
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run-kv-hybrid-state.cmake)
    set_tests_properties(test-kv-hybrid-state-offload PROPERTIES LABELS "model" SKIP_REGULAR_EXPRESSION "KV_HYBRID_SKIP")
endif()

# PXA_QSA_GRID_VERIFY's comparison, as a pure predicate. See the block comment in the test: the
# lever could not pass on the real model because it compared per-ubatch delta fields, and its &&
# ordering meant the grid itself was never reached.
llama_build(test-qsa-grid-verify.cpp)
target_include_directories(test-qsa-grid-verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-qsa-grid-verify)
# PXA_CKPT_EVICT: the prompt-checkpoint eviction policy, against a verbatim copy of the policy it
# replaces. Header-only (examples/server/pxa-ckpt-evict.h), so no server and no model.
llama_build(test-ckpt-evict.cpp)
target_include_directories(test-ckpt-evict PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
llama_test(test-ckpt-evict)
# PXA_CKPT_RAM_v1: context checkpoints and the RAM prompt cache sized from the free RAM (and the
# container limit). Header-only (common/pxa-ckpt-ram.h), so no server and no model.
llama_build(test-ckpt-ram.cpp)
llama_test(test-ckpt-ram)
# PXA_SPEC_ADAPTIVE_LOAD: the load-adaptive speculative depth policy. Header-only
# (examples/server/pxa-spec-load.h), so no server and no model.
llama_build(test-spec-load.cpp)
target_include_directories(test-spec-load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
llama_test(test-spec-load)
# PXA_MTP_DRAFT_CACHE_ONLY: the gate that decides when the MTP head may run the reduced K/V-only
# graph. Header-only (common/pxa-mtp-cache-only.h); the whole safety of the feature is this
# predicate, so the test enumerates its domain exhaustively.
llama_build(test-mtp-cache-only.cpp)
llama_test(test-mtp-cache-only)
# PXA_JIT_FIRST_START_v1 / PXA_JIT_WEDGE_v1: the SASS-vs-JIT verdict for a card and the wedge
# watchdog's stretched limit while the driver JIT-compiles kernels. Header-only, no GPU.
llama_build(test-pxa-jit-wedge.cpp)
llama_test(test-pxa-jit-wedge)
# PXA_TOPK_RAW shadow-compare harness (CPU only, takes a tiny gguf)
llama_build(test-topk-raw-shadow.cpp)

# PXA_REP_PREV_v1 + PXA_REP_DRY_GREEDY_v1 (2026-10-04): repeat / frequency / presence penalties saw a
# ONE-token history and DRY never ran at temperature 0, so every repetition setting was a no-op on a
# looping greedy request. CPU only. The ctest driver generates the tiny qwen4exp fixture with
# tests/gen_tiny_qwen4exp.py into the build tree and runs the test twice (PXA_TOPK_RAW=0 and =1: the
# default path and the fast path must agree). Reported skipped where python3 or numpy is missing.
llama_build(test-rep-penalty-window.cpp)
add_test(NAME test-rep-penalty-window
    COMMAND ${CMAKE_COMMAND}
        -DPYTHON=${Python3_EXECUTABLE}
        -DGEN=${CMAKE_CURRENT_SOURCE_DIR}/gen_tiny_qwen4exp.py
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/rep-penalty-window-fixtures
        -DEXE=$<TARGET_FILE:test-rep-penalty-window>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run-rep-penalty-window.cmake)
set_tests_properties(test-rep-penalty-window PROPERTIES LABELS "model" SKIP_REGULAR_EXPRESSION "REP_PENALTY_SKIP")
# PXA_REP_GUARD scope: which weight mixes arm the guard (PXQ1 any size, PXQN1 by byte share). Header-only.
llama_build(test-rep-guard-scope.cpp)
target_include_directories(test-rep-guard-scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-rep-guard-scope)

# PXQ x Q8 integer dot accuracy/agreement harness (CPU only, no model; needs AVX2/FMA/F16C, x86).
# pxq-cpu.c and pxq-dot.c are already compiled into libggml (see ggml/src/CMakeLists.txt), so
# this only needs their headers on the include path plus a link against ggml for the object code.
# PXA_GLM5NEXT: KDA (per-channel forget gate) vs an independent reference, CPU only.
llama_build(test-kda-delta-net.cpp)
target_link_libraries(test-kda-delta-net PRIVATE ggml)
llama_test(test-kda-delta-net)

# GGML_OP_REDUCE, the -sm graph all-reduce primitive. Until 2026-09-08 this op had no CPU
# implementation and no test of any kind, which is how a graph-construction defect in the
# -sm graph path went unnoticed for a month. CPU only, no GPU, no model.
llama_build(test-reduce-split.cpp)
target_link_libraries(test-reduce-split PRIVATE ggml)
llama_test(test-reduce-split)

# PXA_XCACHE_SKIP_COLD_RANK: the split-ids op that drops the cold tail of a decode / verify window,
# against an independent reference (lever off = the plain split; hot ids untouched; merge 0 + no renormalisation). CPU always, no
# model; with a CUDA build and a visible device also the kernels, the routing counters and the device skip counters.
llama_build(test-pxa-xcache-skip-cold.cpp)
target_link_libraries(test-pxa-xcache-skip-cold PRIVATE ggml)
llama_test(test-pxa-xcache-skip-cold)

# The PXQ K-split, produce-and-compare, one row per tier. Slices a panel-addressed buffer with the
# same function the split uploader calls, dequantises the shards with the CPU panel decoder, and
# requires the result to be BIT-IDENTICAL to the same columns of the unsplit tensor. This is what
# admits a tier to a dim-0 cut. CPU only, no GPU, no model.
llama_build(test-pxq-ksplit.cpp)
target_include_directories(test-pxq-ksplit PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(test-pxq-ksplit PRIVATE ggml)
llama_test(test-pxq-ksplit)
llama_build(test-pxa-tsplit-bounds.cpp)
llama_test(test-pxa-tsplit-bounds)
# PXA_TSPLIT_WORKERS_ADAPT (workers vs plain loop per graph, by measured period) and the default worker cpu pick
# (ggml/src/pxa-tp-adapt.h), table-driven: quiet / collapse / contention arriving and leaving / drift / 3% bar / skip /
# wide graphs / cpuset picks. Pure logic, CPU only, no model.
llama_build(test-pxa-tp-adapt.cpp)
target_include_directories(test-pxa-tp-adapt PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_test(test-pxa-tp-adapt)

# PXA core step 3 (2026-09-25): the engine's own serve-flag defaults over a case table (pure, no GPU),
# and the lever catalog covering every PXA_*/PXQ_* lever in the tree.
llama_build(test-pxa-autoconfig.cpp)
llama_test(test-pxa-autoconfig)
# The CUDA graph OOM guard headroom-shed rule (ggml-cuda/pxa/pxa-cgraph-shed.h), CPU model
llama_build(test-pxa-cgraph-shed.cpp)
llama_test(test-pxa-cgraph-shed)

llama_build(test-pxa-profile-keep.cpp)
llama_test(test-pxa-profile-keep)
# the KQMMV verify-identity width rule (pure host): PXA_KQMMV_VERIFY_ID_SMALL is a switch, _W the width (option A)
llama_build(test-pxa-kqmmv-vid.cpp)
llama_test(test-pxa-kqmmv-vid)

# where the expert cache looks for its routing counts (src/llama-pxa-counts.h): every name the model answers to, and a
# set-but-unusable PXA_XCACHE_COUNTS treated as unset (2026-10-12)
llama_build(test-pxa-xcache-counts.cpp)
llama_test(test-pxa-xcache-counts)
# PXA_XCACHE_OT (glm-mc-2): the expert split under a user -ot placement
llama_build(test-pxa-xcache-ot.cpp)
llama_test(test-pxa-xcache-ot)
# PXA_XCACHE_ASYNC: the cold-path host worker against the same nodes as a plain CPU graph, byte for byte (host driver, no GPU;
# tests/run-xcache-async-test.sh --gpu N drives the real submit / wait kernels). --clamp: the clamped SwiGLU chain of DeepSeek-V4 /
# GLM-5.x (PXA_XCACHE_ASYNC_CLAMP, the GLM-5.x work mechanism M1); a clamp-less worker against that reference mismatches 49 of 57 cases.
llama_build(test-pxa-xcache-async.cpp)
target_include_directories(test-pxa-xcache-async PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
llama_test(test-pxa-xcache-async NAME test-pxa-xcache-async ARGS --cases 200)
llama_test(test-pxa-xcache-async NAME test-pxa-xcache-async-clamp ARGS --clamp 0.5 --cases 200)
llama_test(test-pxa-xcache-async NAME test-pxa-xcache-async-gateless ARGS --gateless --cases 200)
# N1: an unregistered width (or a failed CPU sub-graph) aborts the process; the worker never hands the card zero rows
llama_test(test-pxa-xcache-async NAME test-pxa-xcache-async-fatal ARGS --expect-fatal)
# not a ctest: needs the tiny glm5next GGUF; driven by tests/run-xcache-ot-bitid.sh
llama_build(test-pxa-xcache-ot-logits.cpp)
find_program(PXA_PYTHON3 python3)
if (PXA_PYTHON3)
    add_test(NAME test-pxa-lever-catalog
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/scripts/pxa-lever-catalog.py --check)
    # every ggml-pxqn.def export is defined in a file the MSVC build compiles (else the Windows dll fails to link: LNK2001)
    add_test(NAME test-pxqn-def-exports
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/scripts/pxa-def-exports-check.py ${PROJECT_SOURCE_DIR})
    # bug #344: a refused MTP draft context falls back to plain decoding (tiny qwen4exp + NextN block on the CPU; needs numpy)
    if (LLAMA_BUILD_SERVER AND PXA_BASH AND NOT WIN32)
        add_test(NAME test-mtp-init-fallback
                 COMMAND ${CMAKE_COMMAND} -E env PXA_LLAMA_SERVER=$<TARGET_FILE:llama-server> CUDA_VISIBLE_DEVICES=
                         ${PXA_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/mtp-init-fallback-tiny-test.sh 18796)
        set_tests_properties(test-mtp-init-fallback PROPERTIES TIMEOUT 600)
    endif()
    # install.sh step 2: the GPU check accepts every arch the release builds carry (sm_60/61/70/75)
    if (NOT WIN32)
        add_test(NAME test-install-caps
                 COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-install-caps.py)
    endif()
    # every fatbin in this build's CUDA binaries carries SASS for every real arch of the build (scripts/gate-sm75-check.sh
    # is the release package gate: one .cu built with a narrower list would run its older-arch cubin on a newer card while
    # the host side decides for the newer one). Real archs only: "NN-virtual", "native" and "all" add no SASS to check.
    if (GGML_CUDA AND DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT WIN32 AND PXA_BASH)
        set(_pxa_real_archs "")
        set(_pxa_ptx_archs "")
        foreach(_a ${CMAKE_CUDA_ARCHITECTURES})
            if (_a MATCHES "^([0-9]+)(-real)?$")
                list(APPEND _pxa_real_archs ${CMAKE_MATCH_1})
            endif()
            if (_a MATCHES "^([0-9]+)(-virtual)?$")   # plain NN embeds PTX too; NN-real does not
                list(APPEND _pxa_ptx_archs ${CMAKE_MATCH_1})
            endif()
        endforeach()
        string(REPLACE ";" " " _pxa_ptx_archs_s "${_pxa_ptx_archs}")
        if (_pxa_real_archs)
            string(REPLACE ";" " " _pxa_real_archs_s "${_pxa_real_archs}")
            set(_pxa_cuo_hints ${CUDAToolkit_BIN_DIR})
            if (CMAKE_CUDA_COMPILER)
                get_filename_component(_pxa_nvcc_dir ${CMAKE_CUDA_COMPILER} DIRECTORY)
                list(APPEND _pxa_cuo_hints ${_pxa_nvcc_dir})
            endif()
            find_program(PXA_CUOBJDUMP cuobjdump HINTS ${_pxa_cuo_hints})
            set(_pxa_cuo_flag --no-cuobjdump)
            if (PXA_CUOBJDUMP)
                get_filename_component(_pxa_cuo_dir ${PXA_CUOBJDUMP} DIRECTORY)
                set(_pxa_cuo_flag "")
            endif()
            # the shared libraries land next to their targets (ggml/src), not in bin: name them explicitly
            set(_pxa_cuda_bins ${CMAKE_BINARY_DIR}/bin)
            foreach(_t ggml ggml-cuda ggml-pxqn)
                if (TARGET ${_t})
                    get_target_property(_pxa_tt ${_t} TYPE)
                    if (_pxa_tt STREQUAL "SHARED_LIBRARY" OR _pxa_tt STREQUAL "MODULE_LIBRARY")
                        list(APPEND _pxa_cuda_bins $<TARGET_FILE:${_t}>)
                    endif()
                endif()
            endforeach()
            add_test(NAME test-cuda-archs-embedded
                     COMMAND ${PXA_BASH} ${PROJECT_SOURCE_DIR}/scripts/gate-sm75-check.sh --archs "${_pxa_real_archs_s}" "--ptx=${_pxa_ptx_archs_s}"
                             ${_pxa_cuo_flag} ${_pxa_cuda_bins})
            if (PXA_CUOBJDUMP)
                set_tests_properties(test-cuda-archs-embedded PROPERTIES ENVIRONMENT "CUDA_HOME=${_pxa_cuo_dir}/..")
            endif()
            set_tests_properties(test-cuda-archs-embedded PROPERTIES TIMEOUT 600)
        endif()
    endif()
    add_test(NAME test-control-jit-cache-env
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-control-jit-cache-env.py)
    set_tests_properties(test-control-jit-cache-env PROPERTIES TIMEOUT 120)
    add_test(NAME test-make-windows-release
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-make-windows-release.py)
    set_tests_properties(test-make-windows-release PROPERTIES TIMEOUT 300)
    add_test(NAME test-check-win-package
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-check-win-package.py)
    set_tests_properties(test-check-win-package PROPERTIES TIMEOUT 300)
    add_test(NAME test-release-scan-payloads
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-release-scan-payloads.py)
    set_tests_properties(test-release-scan-payloads PROPERTIES TIMEOUT 300)
    # release-scan.py allow rules (@secrets-allow, @archive-allow)
    add_test(NAME test-release-scan-allow
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-release-scan-allow.py)
    # the public source export's evidence-id scrub (scripts/pxa-public-scrub.py, run by pxa-lever-catalog.py --public-filter)
    add_test(NAME test-pxa-public-scrub
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-public-scrub.py)
    # scripts/pxa-closed-files.txt agrees with the closed-library build and the include graph
    if (EXISTS ${PROJECT_SOURCE_DIR}/scripts/pxa-closed-files.txt)
        add_test(NAME test-pxa-closed-files
                 COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-closed-files.py)
    endif()
    # the clean-package scan rules over the built binaries (scripts/release-scan.py is the package gate): no registry or
    # catalog evidence id in a shipped file, and every binary rule of the private block list when PXA_RELEASE_BLOCKLIST
    # names it (environment at test time, or this cache path; the list itself never enters the repository)
    set(PXA_RELEASE_BLOCKLIST "$ENV{PXA_RELEASE_BLOCKLIST}" CACHE FILEPATH "pxa: the private release block list test-pxa-release-strings applies (optional)")
    set(_pxa_rs_args --source-root ${PROJECT_SOURCE_DIR} --build-dir ${CMAKE_BINARY_DIR})
    if (PXA_RELEASE_BUILD)
        list(APPEND _pxa_rs_args --release)
    endif()
    if (PXA_REGISTRY_EVIDENCE)
        list(APPEND _pxa_rs_args --evidence-ids)
    endif()
    if (PXA_RELEASE_BLOCKLIST)
        list(APPEND _pxa_rs_args --blocklist ${PXA_RELEASE_BLOCKLIST})
    endif()
    add_test(NAME test-pxa-release-strings
             COMMAND ${PXA_PYTHON3} -I ${PROJECT_SOURCE_DIR}/tests/test-pxa-release-strings.py ${_pxa_rs_args})
    # 10-22: the scan takes 842-866 s on the 3-arch test tree, so 900 s timed out under build load (FINAL-6 P100 run 900.26 s)
    set_tests_properties(test-pxa-release-strings PROPERTIES TIMEOUT 2400)
    # bug #281: every level-blind default-on CUDA lever is named by the REFERENCE banner
    add_test(NAME test-pxa-reference-audit
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/scripts/pxa-reference-audit.py)
endif()

# showcase lane (2026-09-25): the pure, header-only layer behind `GET /pxa/explain` and the `/pxa`
# page (examples/server/pxa-explain.h) -- card VRAM arithmetic, the active-lever breakdown, and the
# merged JSON schema. CPU only, no GPU, no server, no model.
# PXA_SPEC_FIXED_WIDTH pad rule (examples/server/pxa-spec-fixed-width.h): pure, no model, no GPU.
llama_build(test-pxa-spec-fixed-width.cpp)
target_include_directories(test-pxa-spec-fixed-width PRIVATE ${PROJECT_SOURCE_DIR}/examples/server)
llama_test(test-pxa-spec-fixed-width)

# PXA_FLASHNEXT_V100_SET MTP depth rule (examples/server/pxa-fn-v100-mtp.h): pure, no model, no GPU.
llama_build(test-pxa-fn-v100-mtp.cpp)
target_include_directories(test-pxa-fn-v100-mtp PRIVATE ${PROJECT_SOURCE_DIR}/examples/server)
llama_test(test-pxa-fn-v100-mtp)

# PXA_THREAD_PLACE (default off): the thread plan (pure, header) over this box's layouts and random CPU sets, and the runtime
# (sched_setaffinity read back, the inherited-mask sweep) on this machine's cpuset. No model, no GPU.
llama_build(test-pxa-thread-place.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-pxa-thread-place.cpp)
target_include_directories(test-pxa-thread-place PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-thread-place)

# Nemotron-H expert split cold path auto choice (cpu / zero-copy from the CPU set's load): /proc/stat parsing, own-time subtraction,
# the decision over forced / oversubscribed / busy / niced / unknown cases, and one live sample of this machine. No model, no GPU.
llama_build(test-pxa-cold-auto.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-pxa-cold-auto.cpp)
target_include_directories(test-pxa-cold-auto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-cold-auto)
# bug #339 placement context guard (src/llama-pxa-place.cpp, PXA_PLACE_CTX_GUARD): the planner never plans a layout the context
# refuses or runs out of VRAM in (all-outputs loads: full logits + the head's sliced GEMM pool), capped contexts keep the old plan
# exactly, plans that met the guard are unchanged. Planner math only: no model, no GPU.
llama_build(test-pxa-place-ctx-guard.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-pxa-place.cpp)
target_include_directories(test-pxa-place-ctx-guard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${PROJECT_SOURCE_DIR}/ggml/src)
llama_test(test-pxa-place-ctx-guard)
# R3a PXA_XCACHE_RING_LEND planner half (ring reported, not charged; off / zero-copy = the old plan). No model, no GPU.
llama_build(test-pxa-xcache-ring-lend.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-pxa-place.cpp)
target_include_directories(test-pxa-xcache-ring-lend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${PROJECT_SOURCE_DIR}/ggml/src)
llama_test(test-pxa-xcache-ring-lend)
# R3d PXA_MTP_RESERVE_RECOMPUTE: the MTP chain reserve priced per step (unchanged) and as replay rollback
llama_build(test-pxa-mtp-reserve-recompute.cpp)
target_include_directories(test-pxa-mtp-reserve-recompute PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-mtp-reserve-recompute)
# PXA_PLE_PIPELINE auto scope (copy slots + exactly 4 sm_60 cards), =0 / =1 / PXA_REFERENCE. No model, no GPU.
llama_build(test-pxa-ple-pipe.cpp)
target_include_directories(test-pxa-ple-pipe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-ple-pipe)

# bug #324: the startup "lab knobs set" warning names only what the user set (helper + driver script + launcher marker test)
llama_build(test-pxa-lab-knobs.cpp)
if (NOT WIN32)
    add_test(NAME test-pxa-lab-knobs COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-pxa-lab-knobs.sh $<TARGET_FILE:test-pxa-lab-knobs>)
    set_property(TEST test-pxa-lab-knobs PROPERTY LABELS main)
    if (PXA_PYTHON3)
        add_test(NAME test-pxa-lab-knobs-launcher COMMAND ${PXA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/test-pxa-lab-knobs-launcher.py)
    endif()
endif()

# port of public PR #13: the launcher sends VRAM with SM75 topology hints and leaves a FIT -ub to the engine
if (PXA_PYTHON3)
    add_test(NAME test-pxa-launch-sm75 COMMAND ${PXA_PYTHON3} -I ${CMAKE_CURRENT_SOURCE_DIR}/test-pxa-launch-sm75.py)
endif()

# the Windows VM release build driver: a stale BUILD_RC from the previous run never passes as a new build, the wait ends
if (PXA_PYTHON3 AND NOT WIN32 AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-windows-release-vm-build.py)   # closed: not in the public tree
    add_test(NAME test-windows-release-vm-build COMMAND ${PXA_PYTHON3} -I ${CMAKE_CURRENT_SOURCE_DIR}/test-windows-release-vm-build.py)
    set_tests_properties(test-windows-release-vm-build PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

llama_build(test-pxa-explain.cpp)
target_include_directories(test-pxa-explain PRIVATE ${PROJECT_SOURCE_DIR}/examples/server)
llama_test(test-pxa-explain)

# llama-bench -o json|csv|md|sql: the data stdout stays clean; every other stdout write goes to stderr
# (common/pxa-stdout-guard.h, beta 2 report 2026-10-19). Runs itself as a child with redirected streams.
llama_build(test-pxa-stdout-guard.cpp)
llama_test(test-pxa-stdout-guard WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# tools/pxa-bench.py's own renderer/self-test on recorded sample data -- no network, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-bench-render
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tools/pxa-bench.py --selftest)
endif()

# PXA Control (pxa-launch --gui): HTTP handlers, token auth, lever validation, presets, the engine
# proxy against a stub server, and the seat's start/stop. Fake cards, no GPU, loopback only.
# The Live tab's sampler (class Live, GET /api/live) is covered by the same file; tests/live-gui-check.py is its headless-browser check
# (fake nvidia-smi + fake servers from tests/live_env.py; needs docker and the playwright image, so it is run by hand, not by CTest).
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control.py)
endif()

# PXA Control hardening (the Control hardening pass): the per-session token on loopback, CSRF / DNS rebinding, one cookie per
# port, extra-args drafter rule, docker and download names, connection / token-flood / body limits, secrets in logs and
# answers, security headers and the hashed CSP. Loopback only, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-security
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-security.py)
endif()

# PXA Control as a package (tools/pxa_control/): the API the single file had, the shim, patch forwarding, the layering,
# and the desktop entry of the compiled pxa-control binary (engine discovery, first-run models folder, single instance).
# PXA Control writes no bytecode next to its own sources (review N3): started through the shipped wrapper, the launcher and the
# shim with the switch removed from the environment, the copied install tree holds no __pycache__; every launcher carries it.
if (PXA_PYTHON3 AND NOT WIN32)
    add_test(NAME test-control-no-bytecode
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-control-no-bytecode.py)
endif()

if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-package
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-package.py)
endif()

# PXA Control, bug #373: load progress from the server child's /proc/<pid>/io (bytes, rate, percent/ETA against the device
# buffer sizes), "silent" = no log output AND no read progress, the slow-disk hint. Fake /proc reader, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-load-progress
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-load-progress.py)
endif()

# PXA Control, beta 2 fixes (2026-10-19): the bench's KV cache type (as served by default, shown in the result), Report a
# problem for a server started by hand (found on a port, a listed process/container, or a typed LAN/loopback address), and the
# download's sha256 hashed as it arrives with progress + time left. Loopback only, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-beta2
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-beta2.py)
endif()

# PXA Control's FREE shareable bench card (tools/pxa_control/share.py + share.html/js/css; todo control-share-report): the card
# from a bench-history row, the public text rules (no paths/flags/hashes, no rival engine names), the self-contained SVG,
# the /api/share routes on a real Control, the token guard, and never gated (no Store / entitlement call). No GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-share
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-share.py)
endif()

# PXA Control's mDNS advert (tools/pxa_control/mdns.py, --lan only, PXA_CONTROL_MDNS=0 off): wire format, which questions it
# answers and how, no secret in TXT, interface choice, and a live advertise -> browse round trip on a private port (never 5353). No GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-mdns
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-mdns.py)
endif()

# PXA Control's licensed-library panel at ROUTE level (todo libupdate-ui-test): GET /api/lib/update (?check=1),
# POST apply / rollback / settings on a real Control with the updater swapped for a fake (C.LU = fake): payload shape,
# channel / auto validation, the busy sentence, refusals as 4xx, auto-apply, no-updater build. No network, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-libupdate
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-libupdate.py)
endif()

# PXA Store + plugins, local side (tools/pxa_control/store, plugins): Ed25519 install key (RFC 8032 vector), grant
# verification (signature, prefix, machine binding, times), the entitlement cache (replay, revocation, clock rollback,
# offline grace), the two store routes, manifest validation/signature/file hashes and the consent store. Test keys only.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-store
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-store.py)
endif()

# PXA Control's store client on the spec readings Core's gate and the plugin host share: https only (plain http on loopback),
# a garbled / cut-off reply = offline, a non-JSON 2xx = malformed (nothing changes), owned[] status locks, and the shared
# 8-vector fixture (tests/fixtures/anibal-entitlement-vectors.json, sha256 pinned on all three sides). Fake licd, loopback.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control-store-verify
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control-store-verify.py)
endif()

# PXA Control's Profiles backend (tools/pxa_ctl): validation, clamping, apply + verify + rollback, undo and the safety
# watch, the auto-start supervisor, schedules, lock files / reserved cards, Benchmark mode, the /api/gpu routes and the
# resolved-context helpers. The mock adapter only: no GPU, no nvidia-smi, loopback only.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-ctl
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-ctl.py)
endif()

# PXA Control's history store (v3.1, tools/pxa_telemetry.py): settings, averaging and sums, minute rollups, retention and the
# size cap, bucketed reads, CSV, one writer per file, the Prometheus text. Temp files and a fake clock; no GPU, no network.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-telemetry
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-telemetry.py)
    add_test(NAME test-pxa-v31-gaps
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-v31-gaps.py)
endif()

# PXA Control's Home Assistant MQTT publisher (v3.1, tools/pxa_mqtt.py): the settings and their environment
# overrides with the password masked, the topic root, the packet framing on the wire (the fixed header is ONE
# type byte then a varint length), the discovery payload Home Assistant consumes, the topic shapes, and the
# availability behaviour on both shutdown paths. No broker, no network; a fake socket and a fake snapshot.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-mqtt
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-mqtt.py)
endif()

# ...and the same payloads end to end, on the wire: the real App, its real Live sampler reading this machine's
# /proc/meminfo, the real MQTT client, and a broker socket on loopback that records what arrives. This is what
# catches a UNIT mistake (a MiB count published under a "%", an unrounded float) that a snapshot-shaped unit
# test cannot see, because the mistake is in the units, not in the shape. No broker needed, no network.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-mqtt-live
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/mqtt-e2e-live.py)
    set_tests_properties(test-pxa-mqtt-live PROPERTIES TIMEOUT 120)
endif()

# PXA Control's in-house chat agent (v3.1, tools/pxa_chat): the text tool-call parser, the folder fence, the guards, approvals,
# picker selection, the agent loop (native tools, text protocol, plain chat, cancel, max steps, step timeout, re-attach) and the
# /api/chat routes inside a Control instance, all against a stdlib mock OpenAI server (tests/pxa_chat_mock.py). No GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-chat
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-chat.py)
endif()

# What a release stages must be import-complete: every sibling module a shipped PXA Control tool imports has to be
# named in scripts/make-release-tarball.sh and docker/Dockerfile too. A file that was never named is skipped in
# silence by both, so the package builds and the tool breaks at its first call (tools/pxa_expert_map.py did).
if (PXA_PYTHON3)
    add_test(NAME test-staged-tools
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-staged-tools.py)
endif()

# A path a shipped doc tells the user to mount must be one the image's own user can write. The image runs
# as pxq (home /work) and /root is 0700 root, so /root/.cache/pxa is unreachable inside the container: the
# v3.1 docs, compose file and entrypoint all named it for the learned expert map, which could therefore
# never persist. Reads the tree only.
if (PXA_PYTHON3)
    add_test(NAME test-container-paths
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-container-paths.py)
endif()

# PXA Control's Encode tab: the adapter that is the only code that knows the quantizer's command line, the fit math and tier
# recommendation, the disk / RAM / VRAM / licence checks, the resumable job runner, and the signed "Get the encoder" download.
# Fake encoder, fake engine tools, fake Hugging Face and licence servers on loopback; no GPU. (tests/encode-gui-check.py is the
# headless-browser click-through; it needs docker and the playwright image, so it is run by hand, not by CTest.)
if (PXA_PYTHON3)
    add_test(NAME test-pxa-encode
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-encode.py)
endif()

# The licensed PXQN library as its own updatable artifact (tools/pxa_lib_update.py, driven by `pxa-update lib`): the signed
# release manifest, the download tickets, the staged swap of lib/current that moves BOTH copies at once, and the rollback to
# the previous release or to the library the engine shipped. Fake licence server on loopback, real install tree, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-lib-update
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-lib-update.py)
endif()

# The engine beta channel for Valued Supporters (tools/pxa_engine_update.py, driven by `pxa-update --channel beta`): the signed
# engine release, the role gate, the download check against the signature, the C hand-over (apply-file) and the way back to stable.
# Fake licence server + fake GitHub on loopback, real install tree, no GPU. The C side runs when PXA_UPDATE_BIN is set or cc exists.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-engine-update
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-engine-update.py)
endif()

# The stable updater tools/pxa-update.c against a fake GitHub release on loopback (tests/test-pxa-update.sh): check, apply,
# rollback, checksum and user-data refusals, and the engine tarball pick: the arch-true "-sm<list>" name outranks the legacy
# "-sm60_61_70" alias only when its list runs on the local card (PXA_UPDATE_SM stands in for nvidia-smi), an Ampere-only
# list never lands on a P100/V100. Linux host tool (fork, popen); needs bash, curl and python3. No GPU.
find_program(PXA_CURL curl)
if (UNIX AND NOT APPLE AND PXA_PYTHON3 AND PXA_BASH AND PXA_CURL)
    add_executable(test-pxa-update-bin ${PROJECT_SOURCE_DIR}/tools/pxa-update.c)
    add_test(NAME test-pxa-update
             COMMAND ${PXA_BASH} ${PROJECT_SOURCE_DIR}/tests/test-pxa-update.sh)
    set_tests_properties(test-pxa-update PROPERTIES ENVIRONMENT "BIN=$<TARGET_FILE:test-pxa-update-bin>" TIMEOUT 300)
endif()

# The CPU preflight that picks lib/ (fast) or lib-compat/ (any x86-64 CPU) from /proc/cpuinfo: tools/pxa-cpu-check.sh and its Python
# twin in tools/pxa-launch.py, on cpuinfo fixtures (qemu64, Nehalem, Sandy/Ivy Bridge, Haswell, Zen 4, no flags line). No engine, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-cpu-check
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-cpu-check.py)
endif()

# The per-model thinking switch (tools/pxa_thinking.py): detection, on/off renders of each family's
# template (jinja2 if present), launch flags, the request rewrite, and the Control wiring. No GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-thinking
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-thinking.py)
endif()

# PXQN (ggml-pxqn.h): GGML_OP_PXQN_RHT on the CPU against the explicit H128 definition, the
# pinned sign stream, W'x' == Wx, k0 shards bit-identical, thread/stride invariance. No model.
# The PXQN tests exercise the closed libggml-pxqn (loaded at run time next to libggml); the public tree
# carries neither the library sources nor these tests.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-rht.cpp AND PXA_PXQN_CLOSED_SRC)
    llama_build(test-pxqn-rht.cpp)
    target_link_libraries(test-pxqn-rht PRIVATE ggml)
    add_dependencies(test-pxqn-rht ggml-pxqn)
    llama_test(test-pxqn-rht)
endif()

# PXA Overdrive: the recipe rows, their order and the class/fit decision the closed library answers with through the open
# registry's hook (ggml-pxqn-overdrive.h; the open test-pxa-autoconfig checks only the hooks). Closed tree only.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-overdrive.cpp AND PXA_PXQN_CLOSED_SRC)
    llama_build(test-pxqn-overdrive.cpp)
    add_dependencies(test-pxqn-overdrive ggml-pxqn)
    llama_test(test-pxqn-overdrive)
endif()

# PXA_XCACHE_CPUPOOL_TASK_BARRIER: the cold-expert CPU pool's job/phase scheduler (ggml/src/pxqn-cpupool-sched.h, header-only):
# every task once, phases in order, a preempted worker holds nobody up and never runs a later job's task. Closed tree only.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-cpupool-sched.cpp AND PXA_PXQN_CLOSED_SRC)
    llama_build(test-pxqn-cpupool-sched.cpp)
    llama_test(test-pxqn-cpupool-sched)
endif()

# PXA_PXQN_KS2_SASS: the embedded re-banked sm_60 GEMM cubin (pxqn-ks2-sm60-cubin.h) must be regenerated with
# scripts/pxa-ks2-cubin-gen.sh whenever its sources change; the engine refuses a stale cubin at run time, this makes the drift
# a red ctest. CPU only (a CMake script), closed tree only.
if (PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cuda/pxa/pxqn-ks2-sm60-cubin.h)
    add_test(NAME test-pxqn-ks2-cubin-sync
             COMMAND ${CMAKE_COMMAND} -DROOT=${CMAKE_CURRENT_SOURCE_DIR}/.. -P ${CMAKE_CURRENT_SOURCE_DIR}/pxqn-ks2-cubin-sync.cmake)
endif()

# libggml-pxqn host code is baseline x86-64 with an AVX2 copy of the two CPU-hot entry points chosen at run time
#: the baseline and AVX2 copies of the PXQN row decoder and the RHT must give the same bits.
# Includes ggml/src/pxqn-cpu.c itself (closed tree only). PXQN_REF_LIB=<other libggml-pxqn.so> also compares that
# library's table bit for bit. Linux only: it loads the library with dlopen and compiles pxqn-cpu.c's GCC
# target("avx2") copies, neither of which a native Windows (MSVC) build has.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-cpu-isa.c AND PXA_PXQN_CLOSED_SRC AND NOT WIN32)
    llama_build(test-pxqn-cpu-isa.c)
    target_include_directories(test-pxqn-cpu-isa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    target_compile_options(test-pxqn-cpu-isa PRIVATE -ffp-contract=off)
    target_link_libraries(test-pxqn-cpu-isa PRIVATE ggml ${CMAKE_DL_LIBS} m)
    llama_test(test-pxqn-cpu-isa)
endif()

# PXQN rev 1: the CUDA backend against the CPU backend through real graphs (MUL_MAT at decode and
# prefill widths, FUSED_UP_GATE, PXQN_RHT, determinism, batch invariance). Needs a CUDA device at
# run time, so it is built but not registered with ctest. The kernel-level twin is
# tests/test-pxqn-cuda.cu (standalone nvcc build, see its header).
# PXA_VFUSE: the qwen4exp verify chains as ggml graphs, fused vs unfused byte compare (run it twice)
if (GGML_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxa-vfuse-graph.cpp)
    llama_build(test-pxa-vfuse-graph.cpp)
    target_link_libraries(test-pxa-vfuse-graph PRIVATE ggml)
endif()

if (GGML_CUDA AND PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-backend.cpp)
    llama_build(test-pxqn-backend.cpp)
    target_link_libraries(test-pxqn-backend PRIVATE ggml)
    add_dependencies(test-pxqn-backend ggml-pxqn)
endif()
# PXA_ATTN_TAIL_FUSE + PXA_PXQN_RQ8: fused gated attention tail + producer-side rotated q8_1, levers on
# == off by bits (two child processes). Needs a CUDA device: built, run by hand inside a card lock (not in ctest).
if (GGML_CUDA AND PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-rq8.cpp)
    llama_build(test-pxqn-rq8.cpp)
    target_link_libraries(test-pxqn-rq8 PRIVATE ggml)
    add_dependencies(test-pxqn-rq8 ggml-pxqn)
endif()
# R1 RIX: a routed MoE FFN on the GPU (fused up/gate node) and on the CPU (host-weight decomposition) must be
# bit-equal at widths 1..5, so an expert can move between the GPU and the CPU cold lane without changing an output bit.
# Needs a card: run by hand inside a card lock (PXA_TEST_GPU=1). As a ctest (label gpu) it skips (77) unless PXA_TEST_GPU=1 is set,
# so an unlocked ctest never takes a card.
if (GGML_CUDA AND PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-rix-moe-memcmp.cpp)
    llama_build(test-rix-moe-memcmp.cpp)
    target_link_libraries(test-rix-moe-memcmp PRIVATE ggml)
    add_dependencies(test-rix-moe-memcmp ggml-pxqn)
    add_test(NAME test-rix-moe-memcmp COMMAND test-rix-moe-memcmp)
    set_tests_properties(test-rix-moe-memcmp PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77 ENVIRONMENT "PXA_TEST_UNDER_CTEST=1")
endif()
# the CPU expert kernels (pxqn-cpu-mmv.c, closed): exactness, AVX2 == baseline bits, split / column / routed invariance and the RIX
# mirror + SwiGLU (avx2 == scalar, width 1..8). Built with gcc by tests/run-pxqn-cpu-mmv.sh (the lib's own baseline flags); three
# runs: defaults, the RIX column cache off and the RIX LUT kernel off. x86-64 Linux only.
find_program(PXA_BASH_MMV bash)
if (PXA_PXQN_CLOSED_SRC AND PXA_BASH_MMV AND NOT WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"
    AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-cpu-mmv.c)
    foreach (_v default COLCACHE RIX_LUT)
        add_test(NAME test-pxqn-cpu-mmv-${_v} COMMAND ${PXA_BASH_MMV} ${CMAKE_CURRENT_SOURCE_DIR}/run-pxqn-cpu-mmv.sh test)
        set(_env "W=${PROJECT_SOURCE_DIR}" "O=${CMAKE_CURRENT_BINARY_DIR}/pxqn-cpu-mmv-${_v}")
        if (NOT _v STREQUAL "default")
            list(APPEND _env "PXA_PXQN_${_v}=0")
        endif()
        set_tests_properties(test-pxqn-cpu-mmv-${_v} PROPERTIES ENVIRONMENT "${_env}")
    endforeach()
endif()

# the PXA_XCACHE cold lane (libggml-pxqn cpupool) on synthetic Flash-Next-class stacks: us per step / per distinct cold expert,
# default vs PXA_PXQN_RIX (bench-xcache-lane -rix). CPU only. Its -check mode is a ctest: every step runs with the pool's flow
# mode off and on (PXA_XCACHE_CPUPOOL_FLOW) and the activation + down bytes must match, default and RIX (exit 77 = no pool).
if (PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench-xcache-lane.cpp)
    add_executable(bench-xcache-lane bench-xcache-lane.cpp)
    target_link_libraries(bench-xcache-lane PRIVATE ggml)
    target_include_directories(bench-xcache-lane PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    add_dependencies(bench-xcache-lane ggml-pxqn)
    # no -cpus: the check uses the process's own CPUs (the caller's taskset), never pins outside them
    add_test(NAME test-xcache-lane-flow COMMAND bench-xcache-lane -check -t 4 -steps 24 -ntok 4 -distinct 7)
    add_test(NAME test-xcache-lane-flow-rix COMMAND bench-xcache-lane -check -rix -t 4 -steps 24 -ntok 4 -distinct 7)
    add_test(NAME test-xcache-lane-flow-1tok COMMAND bench-xcache-lane -check -rix -t 3 -steps 24 -ntok 1 -distinct 4)
    # a step wider than PXA_PXQN_RIX_MAXW gives the RIX=0 bytes on the pool's threads too (no mixed bits)
    add_test(NAME test-xcache-lane-rix-maxw COMMAND bench-xcache-lane -check-maxw -rix -t 4 -steps 12 -ntok 4 -distinct 7)
    set_tests_properties(test-xcache-lane-flow test-xcache-lane-flow-rix test-xcache-lane-flow-1tok test-xcache-lane-rix-maxw
                         PROPERTIES SKIP_RETURN_CODE 77)
endif()

# PXA_PXQ_MMQ80: the sm_80+ PXQ4 / PXQN4 prefill GEMM route against its own arithmetic contract (CPU double reference,
# q8-exact activations), a route-served fingerprint per width, NMSE vs the fp32 product, determinism. Needs a card:
# LABEL cuda (ctest -L cuda; CPU-only builds do not build it). Two runs: the shipped default (lever off; below sm_80
# NMSE + determinism only) and the -route twin with PXA_PXQ_MMQ80=1, which SKIPs (77) below sm_80 or without a device.
if (GGML_CUDA AND PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-mmq80.cpp)
    llama_build(test-pxqn-mmq80.cpp)
    target_link_libraries(test-pxqn-mmq80 PRIVATE ggml)
    add_dependencies(test-pxqn-mmq80 ggml-pxqn)
    llama_test(test-pxqn-mmq80 LABEL "cuda")
    set_tests_properties(test-pxqn-mmq80 PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "PXA_PXQ_MMQ80=0")
    llama_test(test-pxqn-mmq80 NAME test-pxqn-mmq80-route ARGS --route LABEL "cuda")
    set_tests_properties(test-pxqn-mmq80-route PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "PXA_PXQ_MMQ80=1")
endif()

# bug #343: a CUDA graph replay must pick up a change of op_params (generic ggml ops; forces capture on sm_60/70
# too). Needs a CUDA device at run time, so it is built but not registered with ctest.
if (GGML_CUDA)
    llama_build(test-cuda-graph-opparams.cpp)
    target_link_libraries(test-cuda-graph-opparams PRIVATE ggml)
endif()

# ggml_l2_norm's eps convention vs the Gated DeltaNet reference eps formula already established
# by GGML_OP_DELTA_NET's own CPU kernel and by test-kda-delta-net.cpp's reference oracle.
# CPU only, no GPU, no model.
llama_build(test-l2-norm-gdn-eps.cpp)
target_link_libraries(test-l2-norm-gdn-eps PRIVATE ggml)
llama_test(test-l2-norm-gdn-eps)

# PXA_RS_RING: the recurrent ring's rollback-by-index must be BYTE-identical to the per-step
# copy path it replaces, for the ssm half and the conv half alike. CPU only, no model.
llama_build(test-rs-ring.cpp)
target_link_libraries(test-rs-ring PRIVATE ggml)
llama_test(test-rs-ring)

llama_build(test-kpool-score.cpp)
target_link_libraries(test-kpool-score PRIVATE ggml)
llama_test(test-kpool-score)

# PXA_FA_F16_KV_CHUNK: the online-softmax merge that folds bounded K/V conversion chunks back
# together. Pure arithmetic, no ggml and no GPU -- so a failure is a merge bug, never a GPU
# nondeterminism story. Pins bit-identity at one chunk and an fp32 round-off bound above it.
llama_build(test-fa-chunk-merge.cpp)
llama_test(test-fa-chunk-merge)

# PXA_MTP_KVPOS_v1: the MTP head's K/V position policy (common/pxa-mtp-kvpos.h) -- the convention
# the prompt warm-up, the draft loop, the accepted-token commit and the draft-region purge all have
# to share. Pure arithmetic over positions, so it runs with no model, no context and no GPU.
llama_build(test-mtp-kvpos.cpp)
llama_test(test-mtp-kvpos)

# PXA_MTP_READBACK_v1: the MTP head's feature row -- the node every head graph tags
# "result_norm" and that mtp_accept_batch() reads back as the next draft step's conditioning
# hidden -- has to be a graph OUTPUT or ggml-alloc reuses its buffer once the head's own
# lm_head has consumed it. No model, no GPU: the tail is built at its real shape and run
# through ggml_gallocr the way llama_decode() allocates a decode graph.
# PXA_MTP_PMIN_TOPK_v1: the MTP confidence floor's compare quantity -- the argmax renormalised
# over a top-k window instead of a full-vocabulary softmax that cannot reach the floor.
llama_build(test-mtp-pmin-topk.cpp)
llama_test(test-mtp-pmin-topk)

llama_build(test-mtp-head-output.cpp)
target_link_libraries(test-mtp-head-output PRIVATE ggml)
llama_test(test-mtp-head-output)

# PXA_MTP_SHORTLIST on every MTP family: a window onto the first N rows of a head is a head of N rows (bit-identical
# logits for every row type), and speculative decoding through a windowed drafter emits exactly plain greedy's tokens for every
# window width and depth. No model, no GPU.
llama_build(test-mtp-shortlist.cpp)
target_link_libraries(test-mtp-shortlist PRIVATE ggml)
llama_test(test-mtp-shortlist)

# Recurrent-state rollback on a rejected draft, per-step checkpoint vs replay (PXA_SPEC_CKPT_REPLAY): the replay of
# k+1 tokens must equal the per-step row k BITWISE on the real CUDA kernel (200 rounds, T 2..9, k 0..T-1), plus the
# 48-layer cost table at T=5. Skips (exit 0) without a CUDA device.
llama_build(test-spec-rollback-cost.cpp)
target_link_libraries(test-spec-rollback-cost PRIVATE ggml)
llama_test(test-spec-rollback-cost)

# PXA_MTP_BATCH_SLOTS: one MTP draft decode per STEP across all drafting slots instead of one chain
# of decodes per slot (common/pxa-mtp-batch-slots.h). The per-sequence half of the draft loop is the
# real header; this replays the serial scheduler and the batched one over it and requires identical
# per-slot drafts, decode counts, stop codes, K/V pairs and purge bounds. No model, no GPU.
llama_build(test-mtp-batch-slots.cpp)
llama_test(test-mtp-batch-slots)

# PXA_MTP_HIDDEN_BY_BATCH_ROW_v1: which row of the target's embedding buffer a caller's index names
# (llama_spec_hidden_row_for_batch_row in src/llama-spec-features.h). Pure integer arithmetic, so
# the whole domain is enumerated -- including the case the old readers answered wrongly and
# silently: a batch row against an output-indexed buffer. No model, no context, no GPU.
llama_build(test-mtp-hidden-rows.cpp)
target_include_directories(test-mtp-hidden-rows PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-mtp-hidden-rows)

# PXA HOT SWAP: the residency-group core (ggml/src/pxa-residency.h) on a fake device -- park frees
# every byte, unpark maps it back at the same address, weights mirrored once, KV every time,
# rollback on a failed unpark. Header-only, CPU only, no model.
llama_build(test-pxa-residency.cpp)
target_include_directories(test-pxa-residency PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
llama_test(test-pxa-residency)
# PXA HOT SWAP: the router (examples/server/pxa-hotswap.h) -- registry parsing, drain-then-swap,
# batching, fairness, failure rollback, cancellation, and a 12-thread invariant stress. Header-only.
llama_build(test-pxa-hotswap.cpp)
target_include_directories(test-pxa-hotswap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
llama_test(test-pxa-hotswap)
# PXA_SPEC_TIMINGS_v1: the rule deciding whether a response reports draft_n / draft_n_accepted,
# shared by the two timings serialisers (examples/server/pxa-spec-timings.h). Header-only.
llama_build(test-spec-timings.cpp)
target_include_directories(test-spec-timings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
llama_test(test-spec-timings)

# PXA_GLM5NEXT: the k-pool grid is compiled straight into the test (it is pure), so this runs
# with no model, no context and no GPU.
llama_build(test-qsa-gather.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-kpool-grid.cpp)
target_include_directories(test-qsa-gather PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test-qsa-gather PRIVATE ggml)
llama_test(test-qsa-gather)

llama_build(test-kpool-cache.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-kpool-grid.cpp)
target_include_directories(test-kpool-cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test-kpool-cache PRIVATE ggml)
llama_test(test-kpool-cache)

llama_build(test-qsa-select.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-kpool-grid.cpp)
target_include_directories(test-qsa-select PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test-qsa-select PRIVATE ggml)
llama_test(test-qsa-select)

# PXA_GLM5NEXT: the short-prompt NaN regression (Part 6 of the port doc). Needs a real model,
# so it is built but NOT registered with ctest; drive it with the tiny synthetic glm5next GGUF
# from tests/gen_tiny_glm5next.py:
#   test-glm5next-short-prompts tiny-glm5next.gguf 16 4
llama_build(test-glm5next-short-prompts.cpp)
# PXA_GLM5NEXT_GRAPH_REUSE: armed decode must be bit-identical to fresh-graph decode (CPU only).
# Needs the tiny GGUF from tests/gen_tiny_glm5next.py, so built but not registered with ctest:
#   test-glm5next-graph-reuse tiny-glm5next.gguf 7 96
llama_build(test-glm5next-graph-reuse.cpp)
# PXA_GLM_CONCAT_I32_CUDA: CPU I32 concat bit-exactness + the CUDA supports_op predicate (CPU only).
llama_build(test-concat-i32.cpp)
target_include_directories(test-concat-i32 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
llama_test(test-concat-i32)

# PXA_GLM5NEXT k-pool: a repeated greedy request must give the same answer after a real-path
# re-reserve (PXA_SCHED_RESERVE_REAL). Needs a k-pool model, so built but NOT registered; drive it
# with the tiny synthetic glm5next GGUF:   test-kpool-reserve-repeat tiny-glm5next.gguf
llama_build(test-kpool-reserve-repeat.cpp)

# PXA_GLM5NEXT: the llama-quantize keep-list, checked against the real tensor inventory.
# Header-only rule (src/pxa-glm5next-quant.h), so this needs src/ on the include path and
# nothing else -- no model, no weights, no GPU.
llama_build(test-glm5next-quant-rules.cpp)
target_include_directories(test-glm5next-quant-rules PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-glm5next-quant-rules)

# PXA_GLM_KDA_FUSE: the CUDA KDA glue-fusion matcher (ggml-cuda/pxa/pxa-glm-kda-match.h, host-only) on
# synthetic copies of the two glm5next chains (mapping bit for bit, mutations decline). CPU only, no model.
llama_build(test-glm-kda-fuse-match.cpp)
target_include_directories(test-glm-kda-fuse-match PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
llama_test(test-glm-kda-fuse-match)
# the GPU half: both chains on CUDA device 0 at decode + prefill widths, outputs to a file; run with PXA_GLM_KDA_FUSE=0 and =1
# and cmp (bit identity of the fused kernels). Needs a GPU, so built but not registered (gpu-pre.sh drives it).
llama_build(test-glm-kda-fuse-gpu.cpp)

# The tiny synthetic glm5next GGUF as a ctest fixture (tests/gen_tiny_glm5next.py: python3 + numpy only), so the
# model-level glm5next gates run in every CPU ctest instead of by hand (the GLM-5.x work):
#   test-glm-kda-fuse-match-tiny  the matcher over the REAL prefill + decode graphs (6 KDA blocks)
#   test-glm5next-graph-reuse     PXA_GLM5NEXT_GRAPH_REUSE: reused-graph decode bit-identical to fresh graphs
#   test-xcache-ot-bitid          PXA_XCACHE_OT: expert split under -ot, raw logits bit-identical to static -ot
if (PXA_PYTHON3)
    execute_process(COMMAND ${PXA_PYTHON3} -c "import numpy" RESULT_VARIABLE _pxa_tiny_np OUTPUT_QUIET ERROR_QUIET)
    if (_pxa_tiny_np EQUAL 0)
        set(_pxa_tiny ${CMAKE_CURRENT_BINARY_DIR}/tiny-glm5next.gguf)
        add_test(NAME glm5next-tiny-gguf
                 COMMAND ${PXA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/gen_tiny_glm5next.py ${_pxa_tiny} --seed 1234)
        set_tests_properties(glm5next-tiny-gguf PROPERTIES FIXTURES_SETUP glm5next_tiny)
        add_test(NAME test-glm-kda-fuse-match-tiny COMMAND $<TARGET_FILE:test-glm-kda-fuse-match> ${_pxa_tiny} 6)
        add_test(NAME test-glm5next-graph-reuse COMMAND $<TARGET_FILE:test-glm5next-graph-reuse> ${_pxa_tiny} 7 96)
        set_tests_properties(test-glm-kda-fuse-match-tiny test-glm5next-graph-reuse PROPERTIES FIXTURES_REQUIRED glm5next_tiny)
        add_test(NAME test-xcache-ot-bitid COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run-xcache-ot-bitid.sh ${CMAKE_BINARY_DIR}
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    else()
        message(STATUS "pxa: python3 has no numpy -- the tiny glm5next model tests are not registered")
    endif()
endif()

# DeepSeek-V4.1 loader (todo dsv41-bringup, G1/G2): arch deepseek41 hparams + per-layer optional tensors on a
# tiny synthetic stub (tests/fixtures/dsv41-stub/gen_stub.py, python3 + numpy), 2 accepted + 9 rejected variants.
llama_build(test-dsv41-load.cpp)
target_include_directories(test-dsv41-load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if (PXA_PYTHON3 AND _pxa_tiny_np EQUAL 0)
    add_test(NAME test-dsv41-load COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-dsv41-load.sh $<TARGET_FILE:test-dsv41-load>)
endif()

# DeepSeek-V4.1 chat template (todo dsv41-bringup, D2): models/templates/deepseek-ai-DeepSeek-V4.1.jinja must render the
# reference encoder's own golden pairs (encoding/tests of the V4.1 release, read in place) byte for byte.
# Skipped (77) when the directory is absent; set PXA_DSV41_ENCODING_TESTS to point at it.
set(PXA_DSV41_ENCODING_TESTS "$ENV{PXA_DSV41_ENCODING_TESTS}" CACHE PATH "DeepSeek-V4.1 encoding/tests directory")
set(_pxa_dsv41_dir "${PXA_DSV41_ENCODING_TESTS}")
if (NOT _pxa_dsv41_dir)   # unset: a path that does not exist, so the golden part reports SKIP (the unit part still runs)
    set(_pxa_dsv41_dir "${CMAKE_CURRENT_BINARY_DIR}/dsv41-encoding-tests-not-set")
endif()
llama_build(test-dsv41-chat.cpp)
add_test(NAME test-dsv41-chat COMMAND $<TARGET_FILE:test-dsv41-chat>
         ${PROJECT_SOURCE_DIR}/models/templates/deepseek-ai-DeepSeek-V4.1.jinja ${_pxa_dsv41_dir})
set_tests_properties(test-dsv41-chat PROPERTIES SKIP_RETURN_CODE 77)

# DeepSeek-V4.1 G5: 384 routed experts x top-6, sqrtsoftplus + selection bias, norm x 1.5, routed SwiGLU via
# mul_mat_id (F32 and Q8_0 experts) vs a double host reference. CPU always; CUDA device 0 too in a CUDA build.
llama_build(test-dsv41-moe.cpp)
target_link_libraries(test-dsv41-moe PRIVATE ggml)
if (GGML_CUDA)
    target_compile_definitions(test-dsv41-moe PRIVATE GGML_USE_CUDA)
endif()
add_test(NAME test-dsv41-moe COMMAND $<TARGET_FILE:test-dsv41-moe>)

# Sinkhorn de-spill (todo glm-sinkhorn-despill): the hc=4 register-resident DSV4/GLM Sinkhorn order vs the
# generic one, bitwise, over random + edge rows, plus the ggml CPU op cross-check. CPU only, no model.
llama_build(test-dsv4-sinkhorn-despill.cpp)
llama_test(test-dsv4-sinkhorn-despill)

# PXQ composition floor accounting (src/pxa-pxq-composition.h): resident-weights-only share, host-side
# tables excluded, on fake tensor lists (todo encoder-composition-floor-ple). Header-only, no model.
llama_build(test-pxa-composition-floor.cpp)
target_include_directories(test-pxa-composition-floor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-composition-floor)

# PXA_RESERVE_GUARD run accounting (src/llama-pxa-reserve-run.h, issue #11): a long-lived server's
# prefill/decode re-reserves never latch the guard; rapid plan switching still does. Pure, no model, no GPU.
llama_build(test-pxa-reserve-run.cpp)
target_include_directories(test-pxa-reserve-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-reserve-run)

# PXA_HEAD_DUAL_LAYER fit guard (src/llama-pxa-head-dual.h): the head copy to the first card is armed only when that card's
# free memory after the weights covers it plus the context still to come. Pure, no model, no GPU.
llama_build(test-pxa-head-dual-fit.cpp)
target_include_directories(test-pxa-head-dual-fit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-pxa-head-dual-fit)

# POLICY_REV 3 tier resolver: name -> tier for both campaign models. Header-only, no llama link.
# The tier policy is part of the PXQ ENCODER (src/pxq-encoder/), so this test exists only in a
# tree that has it.
if (PXA_PXQ_ENCODER)
    llama_build(test-pxq-policy.cpp)
    target_include_directories(test-pxq-policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    llama_test(test-pxq-policy)
endif()
if (PXA_PXQN_ENCODER AND PXA_QUANTIZER_EDITION STREQUAL "owner")
    # PXQN (private): the closed encoder + the CPU reference decoder vs probe.py on real 27B
    # tensors. Needs the reference data of tools/pxqn-probe/mkencref.py (skips without it).
    llama_build(test-pxqn-encoder.cpp)
    target_include_directories(test-pxqn-encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
                                                         ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    target_link_libraries(test-pxqn-encoder PRIVATE ggml)
endif()
if (PXA_PXQN_ENCODER)
    # Pro license tokens (private): sign/verify, expiry + grace, tamper, revocation, wrong key id,
    # clock rollback. Signs its own test tokens with a fixed test key through the system libcrypto.
    llama_build_and_test(test-pxa-license.cpp)
    target_include_directories(test-pxa-license PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()

# Every PXQ tier a file contains must carry its codebook. Shell, stdlib python only; needs a
# built llama-quantize (QUANT=) and an f32 fixture (FIXTURE=, or numpy for the generator).
# Windows cannot exec a .sh: run it through bash (Git for Windows) with the python launcher name
# Windows has, and leave the test out where no bash is found (it is a shell test, not a port gap).
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxq-codebook-kv.sh)
    if (WIN32)
        # Git for Windows' bash only: System32\bash.exe and the WindowsApps alias are the WSL launcher,
        # which runs the script inside a Linux distro where the .exe paths do not resolve.
        find_program(PXA_WIN_GIT_BASH bash
                     HINTS "$ENV{ProgramFiles}/Git/bin" "$ENV{ProgramW6432}/Git/bin" "$ENV{LOCALAPPDATA}/Programs/Git/bin"
                     NO_DEFAULT_PATH)
        if (NOT PXA_WIN_GIT_BASH)
            find_program(PXA_WIN_GIT_BASH bash)
        endif()
        if (PXA_WIN_GIT_BASH MATCHES "[Ss]ystem32|WindowsApps")
            set(PXA_WIN_GIT_BASH "")
        endif()
        # a real interpreter, not the WindowsApps store alias
        find_program(PXA_WIN_PY NAMES python python3 PATHS ENV PATH NO_DEFAULT_PATH)
        if (PXA_WIN_PY MATCHES "WindowsApps")
            set(PXA_WIN_PY "")
        endif()
    endif()
    if (NOT WIN32)
        add_test(NAME test-pxq-codebook-kv
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-pxq-codebook-kv.sh
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
        set_property(TEST test-pxq-codebook-kv PROPERTY ENVIRONMENT
                     "QUANT=$<TARGET_FILE:llama-quantize>")
    elseif (PXA_WIN_GIT_BASH AND PXA_WIN_PY)
        add_test(NAME test-pxq-codebook-kv
                 COMMAND ${PXA_WIN_GIT_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/test-pxq-codebook-kv.sh
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
        set_property(TEST test-pxq-codebook-kv PROPERTY ENVIRONMENT
                     "QUANT=$<TARGET_FILE:llama-quantize>" "PY=${PXA_WIN_PY}")
    else()
        message(STATUS "test-pxq-codebook-kv: no Git bash/python on this Windows host, test not registered")
    endif()
endif()

llama_build(test-pxq-cpu-dot.cpp)
target_include_directories(test-pxq-cpu-dot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
target_link_libraries(test-pxq-cpu-dot PRIVATE ggml)
if (MSVC)
    target_compile_options(test-pxq-cpu-dot PRIVATE /arch:AVX2)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(test-pxq-cpu-dot PRIVATE -mavx2 -mfma -mf16c)
endif()
llama_test(test-pxq-cpu-dot)

# PXQ2/PXQ3 q8_1 MMVQ weight decode. The decode header compiles for the host with PRMT emulated,
# so the whole of it -- the frozen s8 snap, the bit-plane extraction (exhaustively, every one of
# the 2^16 and 2^24 patterns), the panel/slab addressing against the engine's own CPU decoder, and
# the dot's accuracy against the exact path -- is provable without a GPU. CPU only, no model.
llama_build(test-pxq23-mmvq-snap.cpp)
target_include_directories(test-pxq23-mmvq-snap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
target_link_libraries(test-pxq23-mmvq-snap PRIVATE ggml)
llama_test(test-pxq23-mmvq-snap)

if (GGML_CUDA)
    # CUDA is enabled inside ggml/'s directory scope only; tests/ is a sibling, so the
    # language has to be enabled here before a .cu target can be declared.
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA supports_op(DELTA_NET) must only claim the 64/128 head widths the kernel has (other
    # widths fall back to the CPU; before 3456321f68 tests/fixtures/tiny_qwen4exp.gguf aborted).
    # Needs a CUDA context but launches nothing; run by hand inside a GPU lock window, not
    # registered with llama_test(). Prints SKIP without a device.
    add_executable(test-delta-net-cuda-supports test-delta-net-cuda-supports.cpp)
    target_include_directories(test-delta-net-cuda-supports PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_link_libraries(test-delta-net-cuda-supports PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-delta-net-cuda-supports RUNTIME)

    # K8 / K8-C: PXQ wide-store dequant bit-exactness + prefetch-arena race harness.
    # Not registered with llama_test(): it needs a real CUDA device and is run by hand inside
    # the shared GPU lock window.
    add_executable(test-pxq-dq-wide test-pxq-dq-wide.cu)
    target_include_directories(test-pxq-dq-wide PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-pxq-dq-wide PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    set_property(TARGET test-pxq-dq-wide PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    # libggml.so pulls the CUDA driver API (VMM pool); an executable linking it needs the
    # driver library resolved explicitly.
    target_link_libraries(test-pxq-dq-wide PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxq-dq-wide RUNTIME)

    # M1: PXQ4 dp4a MMQ prefill tile. Same treatment as test-pxq-dq-wide -- needs a real CUDA
    # device, run by hand inside the shared GPU lock window, not registered with llama_test().
    add_executable(test-pxq4-mmq test-pxq4-mmq.cu)
    target_include_directories(test-pxq4-mmq PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-pxq4-mmq PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    set_property(TARGET test-pxq4-mmq PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test-pxq4-mmq PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxq4-mmq RUNTIME)

    # LUT-decode GO/NO-GO microbench: incumbent decode mmv vs a memory-only ceiling vs a
    # pair-psum activation LUT, with registers/smem/occupancy per arm. Measurement instrument,
    # not a shipping kernel; needs a real CUDA device, run by hand inside the shared GPU lock
    # window, not registered with llama_test().
    add_executable(test-pxq-lutdec-bench test-pxq-lutdec-bench.cu)
    target_include_directories(test-pxq-lutdec-bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-pxq-lutdec-bench PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    set_property(TARGET test-pxq-lutdec-bench PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test-pxq-lutdec-bench PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxq-lutdec-bench RUNTIME)

    # PXQ MMVQ column-reuse: the pre/post weight-decode-hoist arms of the same kernel, compared
    # bitwise. Same treatment again -- needs a real CUDA device, run by hand inside the shared GPU
    # lock window, not registered with llama_test().
    add_executable(test-pxq-mmvq-cols test-pxq-mmvq-cols.cu)
    target_include_directories(test-pxq-mmvq-cols PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-pxq-mmvq-cols PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    set_property(TARGET test-pxq-mmvq-cols PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test-pxq-mmvq-cols PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxq-mmvq-cols RUNTIME)
    # It was built and installed but never REGISTERED, so it could not fail a gate: a test
    # nobody runs is a test that does not exist. Labelled cuda -- it needs a card.
    llama_test(test-pxq-mmvq-cols LABEL "cuda")

    # PXA_KQMMV: the k-quant / Q8_0 decode GEMV (ggml-cuda/pxa/kq-mmv.cuh) against a double reference and the
    # incumbent q8_1 MMVQ, plus group / batch-invariance / gate-up / determinism bit checks. Needs a card: run
    # by hand inside a GPU window, not registered with llama_test().
    add_executable(test-kq-mmv test-kq-mmv.cu)
    target_include_directories(test-kq-mmv PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-kq-mmv PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    target_compile_options(test-kq-mmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -extended-lambda>)
    set_property(TARGET test-kq-mmv PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test-kq-mmv PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-kq-mmv RUNTIME)

    # PXA_PXQ_MMV_H2: the half2 pair-LUT decode loop for PXQ2/PXQ3. Two zero-tolerance claims (the
    # LUT is the exact fp16 image of the book, and split == unsplit is bitwise inside the mode) plus
    # the error report against the exact fp32 dot32 and the power-of-two guard cases. Needs a card.
    add_executable(test-pxq23-mmv-h2 test-pxq23-mmv-h2.cu)
    target_include_directories(test-pxq23-mmv-h2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-pxq23-mmv-h2 PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    set_property(TARGET test-pxq23-mmv-h2 PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test-pxq23-mmv-h2 PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxq23-mmv-h2 RUNTIME)
    llama_test(test-pxq23-mmv-h2 LABEL "cuda")

    # PXA_FUSE_DELTANET bit2: a fused gather+mask launch the driver REFUSES for memory (a lazily
    # loaded kernel on a full card) is a clean decline to the eager pair, not an abort (bug #207
    # follow-up, ws6-fix). The refusal is injected with PXA_DN_GM_FAULT=1; both modes must be
    # bit-exact. One process per mode: the engine reads its environment once. Needs a card.
    add_executable(test-pxa-dn-gm-refused test-pxa-dn-gm-refused.cpp)
    target_link_libraries(test-pxa-dn-gm-refused PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxa-dn-gm-refused RUNTIME)
    llama_test(test-pxa-dn-gm-refused NAME test-pxa-dn-gm-fused   ARGS fused   LABEL "cuda")
    llama_test(test-pxa-dn-gm-refused NAME test-pxa-dn-gm-refused ARGS refused LABEL "cuda")

    # PXA Nemotron-H phase 2: the in-place Mamba-2 ops (ggml_mamba2_conv + ggml_mamba2_scan) equal the unfused
    # mixer graph bit for bit (gated output and the written-back state rows), decode / prefill / multi-sequence.
    add_executable(test-pxa-mamba2-fuse test-pxa-mamba2-fuse.cpp)
    target_link_libraries(test-pxa-mamba2-fuse PRIVATE ggml)
    install(TARGETS test-pxa-mamba2-fuse RUNTIME)
    llama_test(test-pxa-mamba2-fuse LABEL "cuda")

    # Bug #227: the grouped PXQ prefill GEMM / SCATFUSE-down pickers must honour RAG (K4 ragged-tile
    # skip) on every tier, and RAG on must equal RAG off by bits. Phase 1 is host-only; phase 2
    # needs a card and skips without one.
    add_executable(test-pxq6-ragtail test-pxq6-ragtail.cu)
    target_include_directories(test-pxq6-ragtail PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-pxq6-ragtail PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    set_property(TARGET test-pxq6-ragtail PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test-pxq6-ragtail PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxq6-ragtail RUNTIME)
    llama_test(test-pxq6-ragtail LABEL "cuda")

    # The core lever registry (ggml-cuda/pxa/core/levers.cu) resolves every lever that pxa-enhance.cuh used to
    # resolve inline. Host only, no card: one process per (card set, model profile, environment) over a grid of
    # ~3000 scenarios, each output hashed and compared with tests/pxa-lever-resolve.golden -- which was
    # produced from the tree BEFORE the registry took the resolvers over -- plus the property that a row is
    # resolved on the first read of THAT row. Registered as the DRIVER: every resolver caches per process.
    # Linux only: it links ggml-cuda internals (g_pxa_topology, the resolvers) that a Linux .so exports by default and a
    # Windows DLL does not.
    if (NOT WIN32)
        add_executable(test-pxa-lever-resolve test-pxa-lever-resolve.cu)
        target_include_directories(test-pxa-lever-resolve PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
        target_compile_definitions(test-pxa-lever-resolve PRIVATE
            GGML_USE_CUDA
            GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
            GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
            GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
            GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
            GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
            K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
        set_property(TARGET test-pxa-lever-resolve PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        target_link_libraries(test-pxa-lever-resolve PRIVATE ggml CUDA::cuda_driver)
        install(TARGETS test-pxa-lever-resolve RUNTIME)
        add_test(NAME test-pxa-lever-resolve
                 COMMAND ${PXA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/test-pxa-lever-resolve.py $<TARGET_FILE:test-pxa-lever-resolve>)
        set_tests_properties(test-pxa-lever-resolve PROPERTIES LABELS "cuda")
    endif()

    # port of public PR #13: the SM75 dense selector is the old selector on every cc but 750, and on 750 at the
    # default PXA_TURING_CUBLAS_NE11=64 (every type x ne11 1..4096; host only)
    if (NOT WIN32)
        add_executable(test-pxa-mmq-dense-select test-pxa-mmq-dense-select.cu)
        target_include_directories(test-pxa-mmq-dense-select PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
        target_compile_definitions(test-pxa-mmq-dense-select PRIVATE
            GGML_USE_CUDA
            GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
            GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
            GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
            GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
            GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
            K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
        set_property(TARGET test-pxa-mmq-dense-select PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        target_link_libraries(test-pxa-mmq-dense-select PRIVATE ggml CUDA::cuda_driver)
        add_test(NAME test-pxa-mmq-dense-select COMMAND test-pxa-mmq-dense-select)
        set_tests_properties(test-pxa-mmq-dense-select PROPERTIES LABELS "cuda")
    endif()

    # bug #230: growing the PXQ KSPLIT workspace bumps the allocator generation (so captured CUDA
    # graphs re-capture instead of replaying into the freed buffer). Needs a card.
    add_executable(test-pxq-ws-gen test-pxq-ws-gen.cu)
    target_include_directories(test-pxq-ws-gen PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
    target_compile_definitions(test-pxq-ws-gen PRIVATE
        GGML_USE_CUDA
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
        GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
        GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
        GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
        K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
    set_property(TARGET test-pxq-ws-gen PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test-pxq-ws-gen PRIVATE ggml CUDA::cuda_driver)
    install(TARGETS test-pxq-ws-gen RUNTIME)
    llama_test(test-pxq-ws-gen LABEL "cuda")

    # the wide-warp MoE expert tile GEMMs (pxqn-moe-gemm.cuh, k_pxqn_gemm_id_*_h2w) and the side-stream
    # split against the 8-row kernels, byte for byte, every wide mode, at the 35B-A3B expert shape. Needs a card
    # (SKIP 77 without one); one short pass at ub 512 and one at ub 2048 with heavier routing skew.
    if (PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-moe-wide.cu)  # closed test (pxa-closed-files.txt)
        add_executable(test-pxqn-moe-wide test-pxqn-moe-wide.cu)
        target_include_directories(test-pxqn-moe-wide PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
        target_compile_definitions(test-pxqn-moe-wide PRIVATE
            GGML_USE_CUDA
            GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
            GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD}
            GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X}
            GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y}
            GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
            K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
        set_property(TARGET test-pxqn-moe-wide PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        install(TARGETS test-pxqn-moe-wide RUNTIME)
        llama_test(test-pxqn-moe-wide NAME test-pxqn-moe-wide-t512  ARGS -T 512 -reps 2 LABEL "cuda")
        llama_test(test-pxqn-moe-wide NAME test-pxqn-moe-wide-t2048 ARGS -T 2048 -skew 1.2 -reps 2 LABEL "cuda")
        set_tests_properties(test-pxqn-moe-wide-t512 test-pxqn-moe-wide-t2048 PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # PXA_REDUCE_PINNED_v1: the cross-device REDUCE, route by route, against a single-device sum.
    # Needs two cards; skips cleanly with one. Registered once per route because every route
    # selector is a per-process static. Linux only (NCCL route, weak-symbol probe of libggml).
    if (NOT WIN32)
        add_executable(test-reduce-pinned test-reduce-pinned.cu)
        target_include_directories(test-reduce-pinned PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include)
        target_compile_definitions(test-reduce-pinned PRIVATE GGML_USE_CUDA)
        set_property(TARGET test-reduce-pinned PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        target_link_libraries(test-reduce-pinned PRIVATE ggml CUDA::cuda_driver)
        install(TARGETS test-reduce-pinned RUNTIME)
        llama_test(test-reduce-pinned NAME test-reduce-pinned-intree ARGS intree LABEL "cuda")
        llama_test(test-reduce-pinned NAME test-reduce-pinned-pinned ARGS pinned LABEL "cuda")
        llama_test(test-reduce-pinned NAME test-reduce-pinned-fused-nopeer ARGS fused-nopeer LABEL "cuda")
        llama_test(test-reduce-pinned NAME test-reduce-pinned-p2p-nopeer ARGS p2p-nopeer LABEL "cuda")
        # Bug #206: the all-reduce over EVERY visible device, the only shape the NCCL route serves.
        # nccl-all must add up whatever NCCL does (inside docker's default 64 MB /dev/shm a four-card
        # NCCL group fails, and its status used to be thrown away); nccl-fault forces the failed-group
        # path with PXA_REDUCE_NCCL_FAULT=1, so the demotion and the fallback run on any 2+ card box.
        llama_test(test-reduce-pinned NAME test-reduce-pinned-nccl-all   ARGS nccl-all   LABEL "cuda")
        llama_test(test-reduce-pinned NAME test-reduce-pinned-nccl-fault ARGS nccl-fault LABEL "cuda")
    endif()

    llama_build(test-dsa-attn.cpp)
    target_link_libraries(test-dsa-attn PRIVATE ggml)
    llama_test(test-dsa-attn LABEL "cuda")

    # Bug #224: the fused softmax->top-k router must keep bit-equal experts (ties to the lower
    # index) and never emit a padding-lane id on a < 32-expert router. Needs a card.
    llama_build(test-topk-moe-tie.cpp)
    target_link_libraries(test-topk-moe-tie PRIVATE ggml CUDA::cuda_driver)
    llama_test(test-topk-moe-tie LABEL "cuda")

    # Bugs #235 / #236: flash attention over a quantized K/V cache stays on the GPU on sm_60/61/70
    # (tile fallback when the vec kernel has no instance; sm_70 decode on the vec kernel) and
    # matches a double-precision reference. Needs a card.
    # PXA_P2P_SELFTEST: the startup peer-copy self-test, and the forced-corruption fallback branch.
    # Needs two cards with a peer path; skips (77) otherwise.
    llama_build(test-pxa-p2p-selftest.cpp)
    target_link_libraries(test-pxa-p2p-selftest PRIVATE ggml)
    llama_test(test-pxa-p2p-selftest LABEL "cuda")
    set_tests_properties(test-pxa-p2p-selftest PROPERTIES SKIP_RETURN_CODE 77)

    llama_build(test-fa-route-qkv.cpp)
    target_link_libraries(test-fa-route-qkv PRIVATE ggml CUDA::cuda_driver)
    llama_test(test-fa-route-qkv LABEL "cuda")
    # bug #236: the ROUTE of a quantized-KV decode node on a V100, read from the PXA_CORE_ROUTES
    # census (a separate process per lever value, which resolves once). Skips (77) off sm_70.
    if (NOT WIN32)   # a .sh driver: Windows cannot exec it
        add_test(NAME test-fa-route-qkv-volta
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-fa-route-qkv-volta.sh $<TARGET_FILE:test-fa-route-qkv>)
        set_tests_properties(test-fa-route-qkv-volta PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)
    endif()
    # PXA_FA_QKV_DIRECT: sm_60 narrow attention over a q4_0 / q8_0 cache read in place -- accuracy
    # against a double reference, determinism, and an arm-vs-arm dump/compare. Needs a card.
    llama_build(test-fa-qkv-direct.cpp)
    target_link_libraries(test-fa-qkv-direct PRIVATE ggml CUDA::cuda_driver)

    # PXA_FA_TILE_V2 on the device. The binary dumps every case's raw fp32 output; the driver runs
    # it once per schedule and requires the two dumps to be byte identical. Registered as the
    # DRIVER, not the binary, because the lever resolves once per process.
    llama_build(test-fa-tile-v2-dev.cpp)
    target_link_libraries(test-fa-tile-v2-dev PRIVATE ggml)
    if (NOT WIN32)   # a .sh driver: Windows cannot exec it
        add_test(NAME test-fa-tile-v2-dev
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-fa-tile-v2-dev.sh
                         $<TARGET_FILE:test-fa-tile-v2-dev> ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(test-fa-tile-v2-dev PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)
    endif()

    # PXA_FA_TILE_512 on the device: the 512/512 and 576/512 tile kernel against the same graph
    # computed on the CPU backend -- which is what the engine runs today for those head sizes,
    # because the CUDA backend declines them and the scheduler places the node on the CPU.
    # Registered as the DRIVER: the lever resolves once per process, and the default-OFF reading
    # (every case DECLINED) is half the proof.
    llama_build(test-fa-tile-big-dev.cpp)
    target_link_libraries(test-fa-tile-big-dev PRIVATE ggml)
    if (NOT WIN32)   # a .sh driver: Windows cannot exec it
        add_test(NAME test-fa-tile-big-dev
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-fa-tile-big-dev.sh
                         $<TARGET_FILE:test-fa-tile-big-dev> ${CMAKE_CURRENT_BINARY_DIR})
        # 77 = not an sm_60 / sm_70 card: the lever never engages there; the driver still fails on an abort or an
        # engaged case off the bar before it skips.
        set_tests_properties(test-fa-tile-big-dev PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)
    endif()

    # Head 512/512 flash attention at SHORT KV lengths (1..300 live cells in a 256-padded range),
    # against a double-precision host reference; nodes carry the graph builder's fused-route mark.
    # Exit 77 without a card.
    llama_build(test-fa-d512-short-kv.cpp)
    target_link_libraries(test-fa-d512-short-kv PRIVATE ggml)
    add_test(NAME test-fa-d512-short-kv COMMAND $<TARGET_FILE:test-fa-d512-short-kv>)
    set_tests_properties(test-fa-d512-short-kv PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)

    # PXA_MMID_F16_GUARD (bug #338): MUL_MAT_ID per-expert rows beyond the f16 range must not turn inf/NaN on the
    # dequant + cuBLAS f16 route, and rows that do not overflow must keep their bits. Exit 77 without a card.
    llama_build(test-mmid-f16-guard.cpp)
    target_link_libraries(test-mmid-f16-guard PRIVATE ggml)
    target_include_directories(test-mmid-f16-guard PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)   # pxq-cpu.h
    add_test(NAME test-mmid-f16-guard COMMAND $<TARGET_FILE:test-mmid-f16-guard>)
    set_tests_properties(test-mmid-f16-guard PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)

    # Large heads (576/512 and 320/256 MLA, 512/512) at prefill widths with GQA 4..16, against a
    # double-precision host reference: on sm_75 the 64-column new-MMA tile has no device code, so the
    # host must stay on the 32-column tile there. Exit 77 without a card or below sm_75.
    llama_build(test-fa-wide-batch.cpp)
    target_link_libraries(test-fa-wide-batch PRIVATE ggml)
    add_test(NAME test-fa-wide-batch COMMAND $<TARGET_FILE:test-fa-wide-batch>)
    set_tests_properties(test-fa-wide-batch PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)

    llama_build(test-rope-flipped.cpp)
    target_link_libraries(test-rope-flipped PRIVATE ggml)
    llama_test(test-rope-flipped LABEL "cuda")

    # Narrow-shape kernel gates (PXA_GETROWS_NARROW / PXA_CPY_FASTDIV /
    # PXA_CONCAT_FLAT / PXA_NORM_REGCACHE). Registered WITHOUT the gates set: in
    # that mode it asserts the shipped defaults. The bit-identity proof is a diff
    # of two runs of this binary, gates off and gates on -- see the file header.
    llama_build(test-narrow-kernel-parity.cpp)
    target_link_libraries(test-narrow-kernel-parity PRIVATE ggml)
    llama_test(test-narrow-kernel-parity LABEL "cuda")

    # PXA_MOE_VERIFY2 expert-grouped MoE verify GEMV: NMSE vs a host reference at verify widths 1..5, plus
    # a HASH-EXACT line per fused case; the bit-identity proof is a diff of a lever-off and a lever-on run
    # (see the file header). Exit 77 without a card.
    llama_build(test-moe-verify2.cpp)
    target_link_libraries(test-moe-verify2 PRIVATE ggml)
    add_test(NAME test-moe-verify2 COMMAND $<TARGET_FILE:test-moe-verify2>)
    set_tests_properties(test-moe-verify2 PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)

    # PXA_ARGSORT_SMEM bitonic argsort with shared-memory keys: indices equal a host copy of the network exactly
    # (heavy ties, ASC/DESC/thresh, 2..1024 columns); the identity proof is the HASH-ALL line of a default run and
    # a PXA_ARGSORT_SMEM=0 run. Exit 77 without a card.
    llama_build(test-argsort-smem.cpp)
    target_link_libraries(test-argsort-smem PRIVATE ggml)
    add_test(NAME test-argsort-smem COMMAND $<TARGET_FILE:test-argsort-smem>)
    set_tests_properties(test-argsort-smem PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)
    # PXA_FLINT_GATE_FUSE: the one-launch sigmoid -> top-k -> norm_w router tail equals the op chain bit for bit (probs, full
    # argsort, gathered and normalized weights; ties, saturation, all-zero rows; 32..1000 experts, 1..512 rows) and stays
    # right where the allocator shares its buffers (#354). The lever is default OFF (=1 opt-in): the first test runs
    # the fused path (=1), the second the op chain (=0); both must print the same HASH-ALL. Exit 77 without a card.
    llama_build(test-flint-gate-fuse.cpp)
    target_link_libraries(test-flint-gate-fuse PRIVATE ggml)
    add_test(NAME test-flint-gate-fuse COMMAND $<TARGET_FILE:test-flint-gate-fuse>)
    set_tests_properties(test-flint-gate-fuse PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77 ENVIRONMENT "PXA_FLINT_GATE_FUSE=1")
    add_test(NAME test-flint-gate-fuse-opchain COMMAND $<TARGET_FILE:test-flint-gate-fuse>)
    set_tests_properties(test-flint-gate-fuse-opchain PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77 ENVIRONMENT "PXA_FLINT_GATE_FUSE=0")
endif()

# PXA: CPU MUL_MAT_ID work queue + same-order tiles equal the old per-expert loop bit for bit
# (30B-A3B / Bolt shapes + ragged edges); with a thread count and reps it also times both paths. CPU test: any build.
add_executable(test-pxa-cpu-mmid test-pxa-cpu-mmid.cpp)
target_link_libraries(test-pxa-cpu-mmid PRIVATE ggml)
install(TARGETS test-pxa-cpu-mmid RUNTIME)
llama_test(test-pxa-cpu-mmid)
# 943 s on a quiet box, 1430 s and then a 1500 s (ctest default) timeout at load ~46 under ctest -j 4 (pass 9): the byte-equality
# cases are CPU-bound at 4 threads, so the budget follows the box, not the code
set_tests_properties(test-pxa-cpu-mmid PROPERTIES TIMEOUT 3600)

# PXA: host PXQN routed experts on the CPU MUL_MAT_ID work queue equal the old per-expert library call bit for
# bit, on a real model's experts (needs a GGUF path, so not a ctest case); with -r it is the dev timing harness (any expert type).
add_executable(test-pxqn-cpu-mmid test-pxqn-cpu-mmid.cpp)
target_link_libraries(test-pxqn-cpu-mmid PRIVATE ggml)
install(TARGETS test-pxqn-cpu-mmid RUNTIME)


# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)



# Bug #209: a speculative rollback rewinds the sampler's token state but not its RNG
# (common_sampler_rewind_keep_rng). Plain struct, no model, no GPU.
llama_build(test-spec-rewind-rng.cpp)
llama_test(test-spec-rewind-rng)

# Bug #213: PXA_SPEC_RELAXED's floor compares a normalised post-chain mass (pxa_spec_cand_mass),
# not the chain's unnormalised exp(l - max). Header-only, no model, no GPU.
llama_build(test-spec-relaxed-pmin.cpp)
llama_test(test-spec-relaxed-pmin)

# PXA_SPEC_SAMPLED / PXA_SPEC_BLOCK_VERIFY: the lossless acceptance rules (common/pxa-spec-sampled.h).
# Per-token rule by chi-square against synthetic p/q; block rule by exact enumeration of the emitted
# token stream against the target's own joint law, plus Monte Carlo and a planted-bug control. CPU only.
llama_build(test-spec-sampled.cpp)
llama_test(test-spec-sampled)
llama_build(test-spec-block-verify.cpp)
llama_test(test-spec-block-verify)
set_tests_properties(test-spec-block-verify PROPERTIES TIMEOUT 1800) # 67-258 s on the Linux box

# PXA_SPEC_SAMPLED_GATE_v1: the sampled draft chain's p_min floor / depth cutoff gate on max q, not on
# the drawn token's q (pxa_spec_draft_gate_prob). Exact enumeration of the emitted stream through the
# chain's gating order (TV < 1e-5) and chi-square at N = 1e6, plus the planted old gate. CPU only.
llama_build(test-spec-sampled-gate.cpp)
llama_test(test-spec-sampled-gate)

# PXA_SPEC_STAGE_ORDER_v1: a table stage typed after the MTP / draft-model stage is moved in front of it (the reverse spelling
# was refused); =0 keeps the typed order. Parameter logic only, CPU.
llama_build(test-spec-stage-order.cpp)
llama_test(test-spec-stage-order)

# Bug #228: speculative accept feedback against the VERIFIED draft length (common/pxa-spec-verified.h).
# Header-only, no model, no GPU.
llama_build(test-spec-verified.cpp)
llama_test(test-spec-verified)

# Bugs #209 / #213 / #228 at their call sites on a real model context (CPU is enough): the relaxed
# branch of common_sampler_sample_and_accept_n, the server's save/verify/restore sampler sequence
# (DIST and adaptive-p), and ngram_map_k accept feedback through common_speculative_*.
# Always registered (bug #308 audit): the test writes its own tiny random-weight llama GGUF (--synth,
# no download, no python), so a plain ctest runs it and it cannot skip. Two runs, one process each
# (the spec switches are read once per process): the relaxed rule (PXA_SPEC_SAMPLED=0) and the
# shipped defaults (lossless rule on, relaxed suppressed). -DPXA_SPEC_TEST_MODEL=<small model.gguf>
# adds the same two runs on that real model (label "model", exit 77 = skipped when the file is gone).
llama_build(test-spec-model-callsites.cpp)
llama_test(test-spec-model-callsites NAME test-spec-model-callsites
           ARGS --synth ${CMAKE_CURRENT_BINARY_DIR}/spec-callsites-tiny-relaxed.gguf)
llama_test(test-spec-model-callsites NAME test-spec-model-callsites-defaults
           ARGS --defaults --synth ${CMAKE_CURRENT_BINARY_DIR}/spec-callsites-tiny-defaults.gguf)
set(PXA_SPEC_TEST_MODEL "" CACHE FILEPATH "small GGUF for test-spec-model-callsites")
if (PXA_SPEC_TEST_MODEL)
    llama_test(test-spec-model-callsites NAME test-spec-model-callsites-real ARGS ${PXA_SPEC_TEST_MODEL} LABEL model)
    llama_test(test-spec-model-callsites NAME test-spec-model-callsites-real-defaults ARGS --defaults ${PXA_SPEC_TEST_MODEL} LABEL model)
    set_property(TEST test-spec-model-callsites-real test-spec-model-callsites-real-defaults PROPERTY SKIP_RETURN_CODE 77)
endif()

# Bug #308: n_probs / post_sampling_probs with speculation on are bit-equal to speculation off at every
# position (self-draft n_max 8 and 3, ngram_map_k; CPU). Boots bin/llama-server, so it needs the server
# built and the same -DPXA_SPEC_TEST_MODEL GGUF; exit 77 = skipped (no server binary or model).
if (PXA_SPEC_TEST_MODEL)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_EXECUTABLE)
        add_test(NAME test-spec-server-probs-onoff
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/spec-server-probs-onoff-test.py
                    --server ${CMAKE_BINARY_DIR}/bin/llama-server --model ${PXA_SPEC_TEST_MODEL}
                    --out ${CMAKE_CURRENT_BINARY_DIR}/spec-server-probs-onoff)
        set_tests_properties(test-spec-server-probs-onoff PROPERTIES SKIP_RETURN_CODE 77 LABELS "model" TIMEOUT 1800)
    endif()
endif()
