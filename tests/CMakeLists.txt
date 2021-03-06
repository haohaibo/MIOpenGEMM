################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )


include(CTest)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests COMMENT "Build all tests.")
add_custom_target(examples COMMENT "Build all examples.")
add_dependencies(check tests)

function(add_test_executable TEST_NAME)
    add_executable (${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME} miopengemm)
    if(WIN32)
        add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    else()
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    endif()
    add_dependencies(tests ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
endfunction(add_test_executable)



function(add_example_executable EX_NAME)
    add_executable (${EX_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${EX_NAME} miopengemm)
    add_dependencies(examples ${EX_NAME})
endfunction(add_example_executable)


add_test_executable(smallgeometrytests smallgeometrytests.cpp)


add_example_executable(basicexample basicexample.cpp)
add_example_executable(deepbench deepbench.cpp)
add_example_executable(redirectionexample redirectionexample.cpp)
add_example_executable(devtest devtest.cpp)
add_example_executable(experiment1 experiment1.cpp)
add_example_executable(initialisationdemo initialisationdemo.cpp)
add_example_executable(gencache gencache.cpp)
add_example_executable(benchthecache benchthecache.cpp)
