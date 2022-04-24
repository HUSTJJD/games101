cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(last LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    /usr/local/cuda/include/
    ${CMAKE_SOURCE_DIR}/headers/
)

link_directories(
    /usr/lib
    /usr/local/lib
    /usr/local/cuda/lib64/
)


enable_language(CUDA)


SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -use_fast_math")
LIST(APPEND CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --generate-code arch=compute_80,code=sm_80 -g -G")

file ( GLOB_RECURSE  root_header_files "src/*.h" )
file ( GLOB_RECURSE  root_head_files "src/*.hpp" )
file ( GLOB_RECURSE  root_cuhead_files "src/*.cuh" )
file ( GLOB_RECURSE  root_src_files "src/*.cpp" )
file ( GLOB_RECURSE  root_cuda_files "src/*.cu" )

list ( APPEND PRJ_HEADER_LIST ${root_header_files} )
list ( APPEND PRJ_SRC_LIST ${root_head_files} )
list ( APPEND PRJ_SRC_LIST ${root_cuhead_files} )
list ( APPEND PRJ_SRC_LIST ${root_src_files} )
list ( APPEND PRJ_SRC_LIST ${root_cuda_files} )


add_executable( ${PROJECT_NAME} ${PRJ_SRC_LIST} ${PRJ_HEADER_LIST})

target_include_directories ( ${PROJECT_NAME}
	PRIVATE
		${PRJ_INCLUDE_DIRS}
)

target_compile_features( ${PROJECT_NAME} 
	PUBLIC 
		cuda_std_11
		cxx_std_11
)

set_target_properties (
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON )


target_compile_options ( ${PROJECT_NAME}
	PRIVATE
		${PRJ_COMPILE_OPTIONS}
)

target_link_libraries(
    ${PROJECT_NAME}
    -lglfw
    -lGL 
    -lX11 
    -lpthread 
    -lXrandr 
    -lXi 
    -lXinerama 
    -lXcursor
    -lcudadevrt
    -lcudart
    -lcurand 
    -lcublas
    ${CUDA_cudadevrt_LIBRARY}
)