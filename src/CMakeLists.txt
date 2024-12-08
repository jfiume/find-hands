cmake_minimum_required(VERSION 3.0.0)
project(find_hands)

find_package(OpenCV 4.5.2 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
