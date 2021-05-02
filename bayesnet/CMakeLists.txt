################################
# BayesNet library
################################
add_library(BayesNet  "factor.h" "factor.cpp" "utils.h" "utils.cpp" "influence_diagram.h" "influence_diagram.cpp" "clique_tree.h" "clique_tree.cpp")
target_include_directories (BayesNet PUBLIC ${CMAKE_SOURCE_DIR})