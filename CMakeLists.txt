cmake_minimum_required(VERSION 3.18)

project(
    thermoplume
    LANGUAGES CXX
)

find_package(
    pybind11
    CONFIG
    REQUIRED
)

pybind11_add_module(
    _thermoplume_cpp
    cpp/src/bindings.cpp
    cpp/src/numerical.cpp
    cpp/src/thermodynamics.cpp
)

target_include_directories(
    _thermoplume_cpp
    PRIVATE
    cpp/include
)

target_compile_features(
    _thermoplume_cpp
    PRIVATE
    cxx_std_17
)

install(
    TARGETS _thermoplume_cpp
    DESTINATION thermoplume
)