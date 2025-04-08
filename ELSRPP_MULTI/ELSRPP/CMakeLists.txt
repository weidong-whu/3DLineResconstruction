# 检查是否已经定义了 CMAKE_TOOLCHAIN_FILE
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/weixi/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")
endif()

cmake_minimum_required(VERSION 3.15)

project(ELSRPP_LIB)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")


find_package(NLopt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem serialization)

# 如果 TCLAP_INCLUDE_DIR 没有找到，手动指定路径
find_path(TCLAP_INCLUDE_DIR tclap/CmdLine.h
    PATHS C:/Users/weixi/tclap-1.4/include
    NO_DEFAULT_PATH
)


# 递归搜索 src/ 下所有 .cpp 文件
file(GLOB_RECURSE ELSRPP_source_list CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# 可选：递归搜索 include/ 下所有 .h 文件
file(GLOB_RECURSE ELSRPP_header_list CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)


add_library(ELSRPP
    ${ELSRPP_header_list}
    ${ELSRPP_source_list}
)

# 设置 include 路径，确保外层能引用
target_include_directories(ELSRPP PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TCLAP_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${NLOPT_INCLUDE_DIRS}
)

# 链接依赖库
target_link_libraries(ELSRPP
    ${OpenCV_LIBRARIES}
    debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
    optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
    debug ${Boost_SERIALIZATION_LIBRARY_DEBUG}
    optimized ${Boost_SERIALIZATION_LIBRARY_RELEASE}
    NLopt::nlopt
)
set_property(TARGET ELSRPP PROPERTY CXX_STANDARD 17)
