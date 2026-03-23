cmake_minimum_required(VERSION 4.0)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)
set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic")

find_package(SFML 3 COMPONENTS Graphics REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(NeuralNetwork main.cpp
        src/Neuron.cpp
        src/Neuron.h
        src/NeuralNetwork.cpp
        src/NeuralNetwork.h
        src/Visualizer.cpp
        src/Visualizer.h
        src/DataLoader.cpp
        src/DataLoader.h)

target_link_libraries(NeuralNetwork SFML::Graphics CURL::libcurl nlohmann_json::nlohmann_json)

add_custom_command(TARGET NeuralNetwork POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data
        $<TARGET_FILE_DIR:NeuralNetwork>/data
)