################################
# Traffic Jam example
################################
add_executable( TrafficJam_Variable_Elimination "traffic_jam_variable_elimination.cpp" "../example_utils.h")
target_link_libraries(TrafficJam_Variable_Elimination BayesNet)