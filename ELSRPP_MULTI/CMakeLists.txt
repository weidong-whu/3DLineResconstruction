# 设置 vcpkg toolchain
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/weixi/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()
cmake_minimum_required(VERSION 3.15)
project(MySolution)



# 设置 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 子目录（静态库 + 主程序）
add_subdirectory(ELSRPP)
add_subdirectory(visualSFM)
