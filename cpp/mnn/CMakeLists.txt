cmake_minimum_required(VERSION 2.8.3)
project(RetinaFace)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/3rd_party/mnn/include
  ${CMAKE_SOURCE_DIR}/3rd_party/opencv-4.4.0/include
)

link_directories(
  ${CMAKE_SOURCE_DIR}/3rd_party/mnn/lib
  ${CMAKE_SOURCE_DIR}/3rd_party/opencv-4.4.0/lib
)

link_libraries(
  MNN
  MNN_Express
  opencv_core
  opencv_highgui
  opencv_imgproc
  opencv_imgcodecs
)

set(SOURCES
  facedet.cpp
  # main.cpp
  src/retinaface.cpp
)

add_executable(facedet ${SOURCES})