cmake_minimum_required(VERSION 3.15)

# Set project name
project(ELSRPP_LIB)

# Use static runtime library (MT for Release, MTd for Debug)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set toolchain file for vcpkg if not already defined
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/weixi/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")
endif()

# Find required packages
find_package(NLopt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem serialization)
find_package(OpenMP REQUIRED)

# Find TCLAP (command-line parser library)
find_path(TCLAP_INCLUDE_DIR tclap/CmdLine.h
    PATHS C:/Users/weixi/tclap-1.4/include
    NO_DEFAULT_PATH
)

# Recursively find all .cpp files under src/
file(GLOB_RECURSE ELSRPP_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Optionally, recursively find all header files under include/
file(GLOB_RECURSE ELSRPP_headers CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)

# Create the static library
add_library(ELSRPP ${ELSRPP_sources} ${ELSRPP_headers})

# Set include directories
target_include_directories(ELSRPP PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TCLAP_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${NLOPT_INCLUDE_DIRS}
)

# Link libraries using keyword signature for debug/optimized
target_link_libraries(ELSRPP
    PUBLIC
        debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
        optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
        debug ${Boost_SERIALIZATION_LIBRARY_DEBUG}
        optimized ${Boost_SERIALIZATION_LIBRARY_RELEASE}
        NLopt::nlopt
        ${OpenCV_LIBRARIES}
)

# 如果找到了 OpenMP，也加上：
if(OpenMP_CXX_FOUND)
    target_link_libraries(ELSRPP PUBLIC OpenMP::OpenMP_CXX)
endif()

# ✅ 关键：为 Visual Studio 分组（按目录结构分组）
foreach(_source IN LISTS ELSRPP_sources)
    file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}/src" "${_source}")
    get_filename_component(_source_path "${_source_rel}" PATH)
    source_group("Source Files\\${_source_path}" FILES "${_source}")
endforeach()

foreach(_header IN LISTS ELSRPP_headers)
    file(RELATIVE_PATH _header_rel "${CMAKE_CURRENT_SOURCE_DIR}/include" "${_header}")
    get_filename_component(_header_path "${_header_rel}" PATH)
    source_group("Header Files\\${_header_path}" FILES "${_header}")
endforeach()
