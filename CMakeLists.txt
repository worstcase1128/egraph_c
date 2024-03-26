cmake_minimum_required(VERSION 3.14)
project(EGraph)

set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")

include_directories(src)

add_executable(demo src/demo/demo.cc)
