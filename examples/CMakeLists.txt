################################
# Examples
################################
add_executable( WaterSprinkler_Variable_Elimination "water_sprinkler_variable_elimination.cpp")
target_link_libraries(WaterSprinkler_Variable_Elimination BayesNet)

add_executable( WaterSprinkler_Clique_Tree "water_sprinkler_clique_tree.cpp")
target_link_libraries(WaterSprinkler_Clique_Tree BayesNet)