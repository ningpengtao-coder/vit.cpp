cmake_minimum_required(VERSION 3.12)
project(vit
        DESCRIPTION "Inference Vision Transformer (ViT) in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/staghado/vit.cpp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the compiler flags, modify here if needed!
# MSVC 默认 Release 使用 /O2，且不支持 -O3 / -march=native，这里按编译器区分。
if (NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()


# Additional flags for C++ (the ffast-math is to be used with caution!)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# deps：复用仓库内共享的 packages/ggml（https://github.com/ningpengtao-coder/ggml）。
set(VIT_GGML_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml"
    CACHE PATH "复用的 ggml 源码目录")
if (NOT EXISTS "${VIT_GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "未找到 ggml 源码目录：${VIT_GGML_SOURCE_DIR}\n"
        "请先执行 git submodule update --init packages/ggml，或通过 -DVIT_GGML_SOURCE_DIR=... 指定。")
endif()

add_subdirectory(${VIT_GGML_SOURCE_DIR} ${CMAKE_BINARY_DIR}/ggml)

set(VIT_GGML_INCLUDE_DIRS
    "${VIT_GGML_SOURCE_DIR}/include"
    "${VIT_GGML_SOURCE_DIR}/examples")

# vit executable
add_executable(${PROJECT_NAME} main.cpp vit.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC ggml)
target_include_directories(${PROJECT_NAME} PUBLIC . ${VIT_GGML_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# quantize executable
option(BUILD_QUANTIZE "Build the quantization executable" ON)

if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC . ${VIT_GGML_INCLUDE_DIRS})
    target_compile_features(quantize PUBLIC cxx_std_11)
endif()

# benchmark executable (optional)
option(BUILD_BENCHMARK "Build the benchmark executable" OFF)

if(BUILD_BENCHMARK)
    find_package(nlohmann_json 3.2.0 REQUIRED)  # JSON package needed for benchmark

    # JSON package needed for benchmark
    add_executable(benchmark tests/benchmark.cpp vit.cpp)
    target_link_libraries(benchmark PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(benchmark PUBLIC ggml)
    target_include_directories(benchmark PUBLIC . ${VIT_GGML_INCLUDE_DIRS})
    target_compile_features(benchmark PUBLIC cxx_std_11)
endif()

# vit C ABI 共享库：供 Dart FFI / ctypes 等外部调用。
option(VIT_BUILD_C_API "Build the vit C ABI shared library" ON)

if (VIT_BUILD_C_API)
    add_library(vit_c SHARED vit.cpp src/vit_c.cpp src/vit_c_bridge.cpp)
    target_link_libraries(vit_c PUBLIC ggml)
    target_include_directories(vit_c PUBLIC . include ${VIT_GGML_INCLUDE_DIRS} PRIVATE src)
    target_compile_features(vit_c PUBLIC cxx_std_11)
    target_compile_definitions(vit_c PRIVATE VIT_C_BUILD VIT_C_SHARED)

    if (MSVC)
        # 同时导出上游 C++ 符号，便于 tests/vit_c_smoke.cpp 复用图像预处理。
        set_target_properties(vit_c PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()

    option(VIT_BUILD_SMOKE "Build the vit C ABI smoke test" OFF)

    if (VIT_BUILD_SMOKE)
        add_executable(vit_c_smoke tests/vit_c_smoke.cpp)
        target_link_libraries(vit_c_smoke PRIVATE vit_c ggml)
        target_include_directories(vit_c_smoke PRIVATE . include ${VIT_GGML_INCLUDE_DIRS})
        target_compile_features(vit_c_smoke PRIVATE cxx_std_11)
    endif()
endif()