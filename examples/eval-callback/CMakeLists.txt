set(TARGET llama-eval-callback)
add_executable(${TARGET} eval-callback.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# test fixture writer (not installed): a tiny random-weight llama with the in-tree llama-spm vocab, so
# test-eval-callback runs offline instead of downloading a model from the Hugging Face hub
add_executable(llama-eval-callback-fixture tiny-llama-fixture.cpp)
target_link_libraries(llama-eval-callback-fixture PRIVATE ggml)
target_compile_features(llama-eval-callback-fixture PRIVATE cxx_std_11)

set(TEST_TARGET test-eval-callback)
add_test(NAME ${TEST_TARGET}
    COMMAND ${CMAKE_COMMAND}
        -DFIXTURE=$<TARGET_FILE:llama-eval-callback-fixture>
        -DEXE=$<TARGET_FILE:${TARGET}>
        -DVOCAB=${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/tiny-eval-callback.gguf
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run-test-eval-callback.cmake)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS eval-callback)
