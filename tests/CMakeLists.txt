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
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

if (LLAMA_LLGUIDANCE)
    llama_target_and_test(test-grammar-llguidance.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
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
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp )
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
llama_build(test-kv-seq-shadow.cpp)
target_include_directories(test-kv-seq-shadow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-kv-seq-shadow)
llama_build(test-kv-mask-parity.cpp)
target_link_libraries(test-kv-mask-parity PRIVATE ggml)
# KV BAND ALIGNMENT: a sequence's logits must not depend on the cell its band starts at. Builds its
# own tiny dense model, so it is hermetic. CPU only, no model file, no card.
llama_build(test-kv-band-align.cpp)
target_link_libraries(test-kv-band-align PRIVATE ggml)
llama_test(test-kv-band-align)
# The same property one level down: the tile flash-attention kernel's running state must be fp32,
# or the 64-cell tile grouping (which is a function of the CELL index) reassociates a 20k-term
# accumulation differently at every band offset. Mirrors the kernel's arithmetic. CPU only.
llama_build(test-fa-band-align.cpp)
llama_test(test-fa-band-align)

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
# Hybrid KV / recurrent-state regressions from the 2026-09-25 bug hunt (#210 #214 #218 #223 #231
# #267): partial restore at np2, defrag vs the state-row permutation and the V layout, no replayed
# ubatches, gpu-fallback at np2, and the qwen4exp PLE window across a speculative rollback. CPU
# only. The ctest driver generates the fixtures with tests/gen_tiny_qwen4exp.py (plain, and --ple)
# into the build tree and runs
#     test-kv-hybrid-state tiny-qwen4exp.gguf tiny-qwen4exp-ple.gguf
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
# PXA_TOPK_RAW shadow-compare harness (CPU only, takes a tiny gguf)
llama_build(test-topk-raw-shadow.cpp)

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

# PXA core step 3 (2026-09-25): the engine's own serve-flag defaults over a case table (pure, no GPU),
# and the lever catalog covering every PXA_*/PXQ_* lever in the tree.
llama_build(test-pxa-autoconfig.cpp)
llama_test(test-pxa-autoconfig)
find_program(PXA_PYTHON3 python3)
if (PXA_PYTHON3)
    add_test(NAME test-pxa-lever-catalog
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/scripts/pxa-lever-catalog.py --check)
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

llama_build(test-pxa-explain.cpp)
target_include_directories(test-pxa-explain PRIVATE ${PROJECT_SOURCE_DIR}/examples/server)
llama_test(test-pxa-explain)

# tools/pxa-bench.py's own renderer/self-test on recorded sample data -- no network, no GPU.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-bench-render
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tools/pxa-bench.py --selftest)
endif()

# PXA Control (pxa-launch --gui): HTTP handlers, token auth, lever validation, presets, the engine
# proxy against a stub server, and the seat's start/stop. Fake cards, no GPU, loopback only.
if (PXA_PYTHON3)
    add_test(NAME test-pxa-control
             COMMAND ${PXA_PYTHON3} ${PROJECT_SOURCE_DIR}/tests/test-pxa-control.py)
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

# PXQN rev 1: the CUDA backend against the CPU backend through real graphs (MUL_MAT at decode and
# prefill widths, FUSED_UP_GATE, PXQN_RHT, determinism, batch invariance). Needs a CUDA device at
# run time, so it is built but not registered with ctest. The kernel-level twin is
# tests/test-pxqn-cuda.cu (standalone nvcc build, see its header).
if (GGML_CUDA AND PXA_PXQN_CLOSED_SRC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxqn-backend.cpp)
    llama_build(test-pxqn-backend.cpp)
    target_link_libraries(test-pxqn-backend PRIVATE ggml)
    add_dependencies(test-pxqn-backend ggml-pxqn)
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

# PXA_GLM5NEXT: the llama-quantize keep-list, checked against the real tensor inventory.
# Header-only rule (src/pxa-glm5next-quant.h), so this needs src/ on the include path and
# nothing else -- no model, no weights, no GPU.
llama_build(test-glm5next-quant-rules.cpp)
target_include_directories(test-glm5next-quant-rules PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
llama_test(test-glm5next-quant-rules)

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
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-pxq-codebook-kv.sh)
    add_test(NAME test-pxq-codebook-kv
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-pxq-codebook-kv.sh
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    set_property(TEST test-pxq-codebook-kv PROPERTY ENVIRONMENT
                 "QUANT=$<TARGET_FILE:llama-quantize>")
endif()

llama_build(test-pxq-cpu-dot.cpp)
target_include_directories(test-pxq-cpu-dot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
target_link_libraries(test-pxq-cpu-dot PRIVATE ggml)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
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

    # PXA_REDUCE_PINNED_v1: the cross-device REDUCE, route by route, against a single-device sum.
    # Needs two cards; skips cleanly with one. Registered once per route because every route
    # selector is a per-process static.
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
    add_test(NAME test-fa-route-qkv-volta
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-fa-route-qkv-volta.sh $<TARGET_FILE:test-fa-route-qkv>)
    set_tests_properties(test-fa-route-qkv-volta PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)
    # PXA_FA_QKV_DIRECT: sm_60 narrow attention over a q4_0 / q8_0 cache read in place -- accuracy
    # against a double reference, determinism, and an arm-vs-arm dump/compare. Needs a card.
    llama_build(test-fa-qkv-direct.cpp)
    target_link_libraries(test-fa-qkv-direct PRIVATE ggml CUDA::cuda_driver)

    # PXA_FA_TILE_V2 on the device. The binary dumps every case's raw fp32 output; the driver runs
    # it once per schedule and requires the two dumps to be byte identical. Registered as the
    # DRIVER, not the binary, because the lever resolves once per process.
    llama_build(test-fa-tile-v2-dev.cpp)
    target_link_libraries(test-fa-tile-v2-dev PRIVATE ggml)
    add_test(NAME test-fa-tile-v2-dev
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-fa-tile-v2-dev.sh
                     $<TARGET_FILE:test-fa-tile-v2-dev> ${CMAKE_CURRENT_BINARY_DIR})
    set_property(TEST test-fa-tile-v2-dev PROPERTY LABELS "cuda")

    # PXA_FA_TILE_512 on the device: the 512/512 and 576/512 tile kernel against the same graph
    # computed on the CPU backend -- which is what the engine runs today for those head sizes,
    # because the CUDA backend declines them and the scheduler places the node on the CPU.
    # Registered as the DRIVER: the lever resolves once per process, and the default-OFF reading
    # (every case DECLINED) is half the proof.
    llama_build(test-fa-tile-big-dev.cpp)
    target_link_libraries(test-fa-tile-big-dev PRIVATE ggml)
    add_test(NAME test-fa-tile-big-dev
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-fa-tile-big-dev.sh
                     $<TARGET_FILE:test-fa-tile-big-dev> ${CMAKE_CURRENT_BINARY_DIR})
    set_property(TEST test-fa-tile-big-dev PROPERTY LABELS "cuda")

    # Head 512/512 flash attention at SHORT KV lengths (1..300 live cells in a 256-padded range),
    # against a double-precision host reference; nodes carry the graph builder's fused-route mark.
    # Exit 77 without a card.
    llama_build(test-fa-d512-short-kv.cpp)
    target_link_libraries(test-fa-d512-short-kv PRIVATE ggml)
    add_test(NAME test-fa-d512-short-kv COMMAND $<TARGET_FILE:test-fa-d512-short-kv>)
    set_tests_properties(test-fa-d512-short-kv PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)

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
endif()


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

# Bug #228: speculative accept feedback against the VERIFIED draft length (common/pxa-spec-verified.h).
# Header-only, no model, no GPU.
llama_build(test-spec-verified.cpp)
llama_test(test-spec-verified)

# Bugs #209 / #213 / #228 at their call sites on a real model context (CPU is enough): the relaxed
# branch of common_sampler_sample_and_accept_n, the server's save/verify/restore sampler sequence
# (DIST and adaptive-p), and ngram_map_k accept feedback through common_speculative_*. Needs a
# GGUF: configure with -DPXA_SPEC_TEST_MODEL=<small model.gguf> to register it (exit 77 = skipped).
llama_build(test-spec-model-callsites.cpp)
set(PXA_SPEC_TEST_MODEL "" CACHE FILEPATH "small GGUF for test-spec-model-callsites")
if (PXA_SPEC_TEST_MODEL)
    llama_test(test-spec-model-callsites ARGS ${PXA_SPEC_TEST_MODEL} LABEL model)
    set_property(TEST test-spec-model-callsites PROPERTY SKIP_RETURN_CODE 77)
endif()
