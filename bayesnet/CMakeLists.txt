################################
# BayesNet library
################################
add_library(BayesNet  "factor.cpp" "utils.cpp" "influence_diagram.cpp" "clique_tree.cpp" "sampling.cpp")
target_include_directories (BayesNet PRIVATE ${CMAKE_SOURCE_DIR})
