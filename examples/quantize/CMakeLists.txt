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
endif()
