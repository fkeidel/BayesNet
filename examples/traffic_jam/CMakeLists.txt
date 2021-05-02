################################
# Traffic Jam example
################################
add_executable( TrafficJam_Variable_Elimination "traffic_jam_variable_elimination.cpp" "${CMAKE_SOURCE_DIR}/examples/example_utils.h")
target_include_directories (TrafficJam_Variable_Elimination PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(TrafficJam_Variable_Elimination BayesNet) 