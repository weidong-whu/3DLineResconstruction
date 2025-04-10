project(pixel4d)
# 检查是否已经定义了 CMAKE_TOOLCHAIN_FILE
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:\Users\weideep\vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")
endif()
add_executable(pixel4d main_pixel4d.cpp)

# ✅ 链接刚刚的静态库 target（名字必须与 ELSRPP 中一致）
target_link_libraries(pixel4d PRIVATE ELSRPP)

# ✅ 如果你要用头文件，也确保能 include
# 这一步是“保险”，其实上面 PUBLIC 已经设置好了
target_include_directories(pixel4d PRIVATE
    ${CMAKE_SOURCE_DIR}/ELSRPP
)
