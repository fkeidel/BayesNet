################################
# Water Sprinkler example
################################
add_executable( WaterSprinkler_Variable_Elimination "water_sprinkler_variable_elimination.cpp")
target_include_directories (WaterSprinkler_Variable_Elimination PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(WaterSprinkler_Variable_Elimination BayesNet)

add_executable( WaterSprinkler_Clique_Tree "water_sprinkler_clique_tree.cpp")
target_include_directories (WaterSprinkler_Clique_Tree PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(WaterSprinkler_Clique_Tree BayesNet)

add_executable( WaterSprinkler_Clique_Tree_Runtime "water_sprinkler_clique_tree_runtime.cpp")
target_include_directories (WaterSprinkler_Clique_Tree_Runtime PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(WaterSprinkler_Clique_Tree_Runtime BayesNet)