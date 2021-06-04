cmake_minimum_required(VERSION 3.5)

project(Darknet_Detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "L:/Project/Mine/Darknet_Detector/thirdparty/opencv-3.4.0/build/installed/share/OpenCV")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(OpenCV_LIBRARIES ${OpenCV_LIBS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${PROJECT_SOURCE_DIR}/../thirdparty/darknet)
link_directories(/usr/local/cuda-10.0/lib64)

FILE(GLOB SC_FILES "*.cpp" "*.h")
add_executable(${PROJECT_NAME} ${SC_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} -ldarknet -lcuda -lcudart -lcudnn -lcurand -lcublas)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
