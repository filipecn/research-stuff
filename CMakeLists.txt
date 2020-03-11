cmake_minimum_required(VERSION 3.7)

# cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
# check for specific machine/compiler options.
include(ConfigCompilerFlags)

project(RESEARCH_STUFF LANGUAGES CUDA CXX)

include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext")
include(ponos)
include(stb)
include(catch2)

add_subdirectory(common)
add_subdirectory(1_smoke_sim)
add_subdirectory(2_openvdb_flip)