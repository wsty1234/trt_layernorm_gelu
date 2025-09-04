cmake_minimum_required(VERSION 3.15)
project(custom_trt_plugins LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# 设置 TensorRT 路径
# 请修改这些路径为你本地 TensorRT 安装
set(TRT_ROOT "/root/TensorRT-10.4.0.26") # 或者 /usr/local/TensorRT
include_directories(${TRT_ROOT}/include)
link_directories(${TRT_ROOT}/lib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(custom_trt_plugins SHARED
    layernorm_plugin.cpp
    gelu_plugin.cpp
    register.cpp
)

target_include_directories(custom_trt_plugins PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(custom_trt_plugins nvinfer nvinfer_plugin cudart)

# 启用 CUDA 源文件
set_source_files_properties(layernorm_plugin.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(gelu_plugin.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(register.cpp PROPERTIES LANGUAGE CUDA)