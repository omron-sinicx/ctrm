cmake_minimum_required(VERSION 3.16)
project(spars_wrapper CXX)

# pybind11
find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++17)

# fcl
find_package(fcl REQUIRED)

# OMPL
find_package(ompl REQUIRED)
include_directories(${OMPL_INCLUDE_DIRS})
link_directories(${OMPL_LIBRARY_DIRS})
add_definitions(${OMPL_DEFINITIONS})

# add src
pybind11_add_module(spars_wrapper ./src/spars.cpp)

target_link_libraries(spars_wrapper PRIVATE ${OMPL_LIBRARIES} fcl)
