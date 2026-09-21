set(TARGET llama-pxq-export)
add_executable(${TARGET} pxq-export.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# pxq-cpu.h (the panel dequant behind --cpu and the --verify cross-check) is internal to ggml,
# so reach it by path rather than exporting it.
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)

if (GGML_CUDA)
    # pxa_pxq_dequant_host lives in libggml behind GGML_CUDA. Without it the tool still works
    # via --cpu; it just cannot use the kernels.
    target_compile_definitions(${TARGET} PRIVATE PXQ_EXPORT_CUDA)
endif()
