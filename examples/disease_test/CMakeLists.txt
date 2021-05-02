################################
# Disease-Test example
################################
add_executable( DiseaseTest "disease_test.cpp")
target_include_directories (DiseaseTest PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(DiseaseTest BayesNet)