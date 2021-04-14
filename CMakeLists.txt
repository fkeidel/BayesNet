# CMakeList.txt : CMake project for BayesNet, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("BayesNet")

add_library(BayesNet  "factor.h" "factor.cpp" "utils.h" "utils.cpp" "influence_diagram.h" "influence_diagram.cpp" "clique_tree.h" "clique_tree.cpp")
target_include_directories (BayesNet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

################################
# GTest
################################
set( gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" )
add_subdirectory(googletest)
enable_testing()
message(STATUS "gtest_SOURCE_DIR : ${gtest_SOURCE_DIR} ")
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################
add_subdirectory(tests)

################################
# Examples
################################
add_subdirectory(examples)