add_executable(test_generalized_eigensolver test_generalized_eigensolver.cpp)
target_link_libraries(test_generalized_eigensolver gtest_main gefast)
add_test(test_generalized_eigensolver test_generalized_eigensolver)
