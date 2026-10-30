set(TARGET llama-quantize)
add_executable(${TARGET} quantize.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
target_compile_definitions(${TARGET} PRIVATE PXA_PXQ_QUANTIZER_URL="${PXA_PXQ_QUANTIZER_URL}")

# `pxq-quantize` — the tool that WRITES PXQ files. Same program and the same CLI; it exists
# under its own name because the encoder ships separately from the runtime. Built only where
# the encoder is built, so a tree without src/pxq-encoder/ simply has no such target.
if (PXA_PXQ_ENCODER)
    target_compile_definitions(${TARGET} PRIVATE PXA_PXQ_ENCODER)

    set(PXQ_TARGET pxq-quantize)
    add_executable(${PXQ_TARGET} quantize.cpp)
    install(TARGETS ${PXQ_TARGET} RUNTIME)
    target_link_libraries(${PXQ_TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${PXQ_TARGET} PRIVATE ../../common)
    target_compile_features(${PXQ_TARGET} PRIVATE cxx_std_11)
    target_compile_definitions(${PXQ_TARGET} PRIVATE PXA_PXQ_ENCODER PXQ_QUANTIZE_TOOL
                               PXA_PXQ_QUANTIZER_URL="${PXA_PXQ_QUANTIZER_URL}")
    target_compile_definitions(${PXQ_TARGET} PRIVATE PXA_QUANTIZER_VERSION="${PXA_QUANTIZER_VERSION}")
    target_compile_definitions(${TARGET}     PRIVATE PXA_QUANTIZER_VERSION="${PXA_QUANTIZER_VERSION}")
    if (PXA_PXQN_ENCODER)
        target_compile_definitions(${TARGET}     PRIVATE PXA_PXQN_ENCODER)
        target_compile_definitions(${PXQ_TARGET} PRIVATE PXA_PXQN_ENCODER)
    endif()
endif()

pxa_add_alias(${TARGET} pxa-quantize)

# pxa-dsv41-convert (PRIVATE: tools/dsv41-convert is a closed path, it links the PXQN codec): DeepSeek-V4.1
# safetensors (MXFP4 experts, FP8 32x32 dense) -> PXQN GGUF, streamed one matrix at a time (no F32 file).
if (PXA_PXQN_ENCODER AND EXISTS ${CMAKE_SOURCE_DIR}/tools/dsv41-convert/dsv41-convert.cpp)
    add_executable(pxa-dsv41-convert ${CMAKE_SOURCE_DIR}/tools/dsv41-convert/dsv41-convert.cpp
                                     ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c)
    target_link_libraries(pxa-dsv41-convert PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(pxa-dsv41-convert PRIVATE ${CMAKE_SOURCE_DIR}/vendor ${CMAKE_SOURCE_DIR}/ggml/include
                                                      ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps)
    target_compile_features(pxa-dsv41-convert PRIVATE cxx_std_17)
endif()
