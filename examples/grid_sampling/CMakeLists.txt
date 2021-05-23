################################
# Grid Sampling example
################################
add_executable( GridSampling "grid_sampling.cpp")
target_include_directories (GridSampling PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(GridSampling BayesNet)