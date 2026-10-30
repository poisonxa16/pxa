# ctest driver for test-eval-callback: write the tiny llama fixture into the build tree (no download),
# then run llama-eval-callback on it and require that the callback printed the forward pass.
#   cmake -DFIXTURE=<llama-eval-callback-fixture> -DEXE=<llama-eval-callback> -DVOCAB=<vocab.gguf>
#         -DOUT=<model.gguf> -P run-test-eval-callback.cmake
execute_process(COMMAND "${FIXTURE}" "${VOCAB}" "${OUT}" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "fixture generation failed (${rc}): ${FIXTURE} ${VOCAB} ${OUT}")
endif()
execute_process(COMMAND "${EXE}" --model "${OUT}" --prompt hello --seed 42 -ngl 0
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message("${out}")
message("${err}")
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "llama-eval-callback failed (${rc})")
endif()
foreach(op "ggml_debug: *inp_embd" "ggml_debug: *result_output")
    if(NOT "${out}${err}" MATCHES "${op}")
        message(FATAL_ERROR "the eval callback never printed '${op}'")
    endif()
endforeach()
