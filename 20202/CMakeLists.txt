cmake_minimum_required( VERSION 3.0)

project(movieFlip)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(movieFlip movieFlip.cpp)

target_link_libraries(movieFlip ${OpenCV_LIBS})