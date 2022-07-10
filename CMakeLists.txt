cmake_minimum_required(VERSION 3.21)
project(Tree)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

find_package(OpenMP REQUIRED)
add_executable(Tree main.cpp Regression_tree.cpp Regression_tree.h Random_forest_tree.cpp Random_forest_tree.h Random_forest_regressor.cpp Random_forest_regressor.h Abstract_regressor.h Tools.cpp Tools.h Table.cpp Table.h)
target_link_libraries(Tree PRIVATE OpenMP::OpenMP_CXX)