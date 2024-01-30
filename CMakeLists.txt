cmake_minimum_required(VERSION 3.27)
project(LearnCuda CUDA CXX)

set(CMAKE_CXX_STANDARD 20 CACHE BOOL "" FORCE)
set(CMAKE_CUDA_STANDARD 20 CACHE BOOL "" FORCE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" FORCE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw EXCLUDE_FROM_ALL)

add_subdirectory(glad EXCLUDE_FROM_ALL)

add_executable(LearnCuda
        Vector.cuh
        Geometry.cuh
        Framebuffer.cu Framebuffer.cuh
        Render.cu Render.cuh
        Viewer.cpp)

target_compile_definitions(LearnCuda PRIVATE GLFW_INCLUDE_NONE)

target_link_libraries(LearnCuda PRIVATE glfw glad)
