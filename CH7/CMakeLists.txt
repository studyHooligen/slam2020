cmake_minimum_required( VERSION 3.0)

project(ch7VO)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")
include_directories("/usr/include/ceres")
include_directories("/usr/local/include")

add_executable(ch7VO feature_extraction.cpp)

target_link_libraries(ch7VO ${OpenCV_LIBS})
target_link_libraries(ch7VO ${CERES_LIBRARIES})
target_link_libraries(ch7VO g2o_core g2o_stuff)