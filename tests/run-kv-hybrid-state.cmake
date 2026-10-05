# ctest driver for test-kv-hybrid-state (bugs #210 #214 #218 #223 #231 #267).
# Generates the two tiny qwen4exp fixtures into the build tree, then runs the test on them.
#   cmake -DPYTHON=<python3> -DGEN=<tests/gen_tiny_qwen4exp.py> -DOUT=<dir> -DEXE=<test binary>
#         -P run-kv-hybrid-state.cmake
# Prints "KV_HYBRID_SKIP" (the test's SKIP_REGULAR_EXPRESSION) when python3 or numpy is
# missing, so a box without them skips rather than fails; any other failure fails the test.
if(NOT PYTHON OR NOT EXISTS "${PYTHON}")
    message(STATUS "KV_HYBRID_SKIP: no python3 interpreter for the fixture generator")
    return()
endif()
execute_process(COMMAND "${PYTHON}" -c "import numpy" RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
if(NOT rc EQUAL 0)
    message(STATUS "KV_HYBRID_SKIP: numpy is not importable by ${PYTHON}")
    return()
endif()
file(MAKE_DIRECTORY "${OUT}")
foreach(args "${OUT}/tiny-qwen4exp.gguf" "${OUT}/tiny-qwen4exp-ple.gguf;--ple")
    execute_process(COMMAND "${PYTHON}" "${GEN}" ${args} RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "fixture generation failed: ${GEN} ${args}")
    endif()
endforeach()
execute_process(COMMAND "${EXE}" "${OUT}/tiny-qwen4exp.gguf" "${OUT}/tiny-qwen4exp-ple.gguf" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "test-kv-hybrid-state failed (${rc})")
endif()
