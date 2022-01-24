cmake_minimum_required(VERSION 3.16)
project(cost_to_go_wrapper CXX)

# pybind11
find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++17)

# add src
pybind11_add_module(cost_to_go_wrapper ./src/cost_to_go.cpp)
