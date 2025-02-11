cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ELSRPP_SRC_PROJ)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ELSD)

# try to find the NLopt headers
find_package(NLopt REQUIRED)

# try to find the eigen3 headers
find_package(Eigen3 REQUIRED)

find_package(CUDA REQUIRED)

find_path(TCLAP_INCLUDE_DIR tclap/CmdLine.h)

# try to find OPENCV
find_package(OpenCV REQUIRED)

# try to find the boost headers
find_package(Boost REQUIRED COMPONENTS filesystem serialization)

if(NOT OpenCV_FOUND)
	message(FATAL_ERROR "Opencv not found.")
endif()

#ELSR++
file( GLOB ELSRPP_header_list ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file( GLOB ELSRPP_source_list ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp) 
file( GLOB ELSRPP_cuda_list ${CMAKE_CURRENT_SOURCE_DIR}/*.cu) 
cuda_add_executable(ELSRPP_SRC
		${ELSRPP_header_list}
		${ELSRPP_cuda_list}
		${ELSRPP_source_list})
target_include_directories(ELSRPP_SRC PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories(ELSRPP_SRC PUBLIC ${OpenCV_INCLUDE_DIRS}
		   ${TCLAP_INCLUDE_DIR}
		   ${EIGEN3_INCLUDE_DIRS}
		   ${NLOPT_INCLUDE_DIRS})
target_link_libraries(ELSRPP_SRC ${OpenCV_LIBRARIES})
target_link_libraries(ELSRPP_SRC ${CUDA_cusolver_LIBRARY})
target_link_libraries(ELSRPP_SRC debug ${Boost_FILESYSTEM_LIBRARY_DEBUG})
target_link_libraries(ELSRPP_SRC optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE})
target_link_libraries(ELSRPP_SRC debug ${Boost_SERIALIZATION_LIBRARY_DEBUG})
target_link_libraries(ELSRPP_SRC optimized ${Boost_SERIALIZATION_LIBRARY_RELEASE})
target_link_libraries(ELSRPP_SRC NLopt::nlopt)
# target_link_libraries(ELSRPP optimized ${Boost_ARCHIVE_LIBRARY_RELEASE})
# target_link_libraries(ELSRPP debug ${Boost_ARCHIVE_LIBRARY_DEBUG})
set_property(TARGET ELSRPP_SRC PROPERTY CXX_STANDARD 17)

