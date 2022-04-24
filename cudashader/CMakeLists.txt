cmake_minimum_required(VERSION 3.18)

project(cudashader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_INCLUDE_CURRENT_DIR ON)


aux_source_directory(. SRC)

add_executable(cudashader  ${SRC})
