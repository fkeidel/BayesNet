################################
# Tutorial
################################
add_executable( Tutorial "tutorial.cpp")
target_include_directories (Tutorial PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(Tutorial BayesNet)