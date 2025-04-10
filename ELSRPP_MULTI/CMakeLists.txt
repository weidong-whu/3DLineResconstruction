# 设置 vcpkg toolchain
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/weixi/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")
endif()
cmake_minimum_required(VERSION 3.15)
project(MySolution)

# Use static runtime library (MT for Release, MTd for Debug)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# 设置 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 子目录（静态库 + 主程序）
add_subdirectory(ELSRPP)
add_subdirectory(visualSFM)
add_subdirectory(pixel4D)