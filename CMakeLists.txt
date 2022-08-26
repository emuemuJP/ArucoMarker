cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
# OpenCVのバージョンを指定
set(OCV_VERSION "4.2")
project(src)
# OpenCVパッケージの設定を検索し読み込む
find_package(OpenCV ${OCV_VERSION} REQUIRED)
# ArUco marker
add_executable(arMarkerGenerator arMarkerGenerator.cpp)

if(OpenCV_FOUND)
    target_include_directories(arMarkerGenerator PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(arMarkerGenerator ${OpenCV_LIBS})
endif()