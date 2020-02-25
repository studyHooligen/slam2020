cmake_minimum_required( VERSION 3.0)

project(imageBasics)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(imageBasics CH5imageBasics.cpp)

target_link_libraries(imageBasics ${OpenCV_LIBS})