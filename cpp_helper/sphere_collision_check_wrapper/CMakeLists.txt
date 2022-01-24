cmake_minimum_required(VERSION 3.16)
project(sphere_collision_check_wrapper CXX)

# pybind11
find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++17)

# add src
pybind11_add_module(sphere_collision_check_wrapper ./src/collision_check.cpp)
