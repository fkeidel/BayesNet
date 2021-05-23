################################
# Unit Tests
################################
add_executable( UnitTests "factor_unit_tests.cpp"  "utils_unit_tests.cpp" "influence_diagram_test.cpp" "test_utils.cpp" "clique_tree_test.cpp" "sampling_test.cpp" "grid_unit_tests.cpp")
target_include_directories (UnitTests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(UnitTests gtest gtest_main BayesNet)
add_test( NAME UnitTests COMMAND UnitTests )
