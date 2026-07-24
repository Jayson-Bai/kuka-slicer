cmake_minimum_required(VERSION 3.19)

project(kuka_prusa_bridge LANGUAGES CXX)

set(PRUSA_SOURCE_DIR "" CACHE PATH "PrusaSlicer source directory")
set(KUKA_NATIVE_OUTPUT_DIR "" CACHE PATH "Directory receiving prusa_bridge.pyd")

if(NOT EXISTS "${PRUSA_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "PRUSA_SOURCE_DIR must point to an extracted PrusaSlicer source tree")
endif()

if(NOT KUKA_NATIVE_OUTPUT_DIR)
    message(FATAL_ERROR "KUKA_NATIVE_OUTPUT_DIR is required")
endif()

# The bridge uses only libslic3r geometry.  GUI, tests, and STEP import do not
# belong to this extension build.
set(SLIC3R_GUI OFF CACHE BOOL "" FORCE)
set(SLIC3R_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLIC3R_ENABLE_FORMAT_STEP OFF CACHE BOOL "" FORCE)

add_subdirectory("${PRUSA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/PrusaSlicer")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(prusa_bridge MODULE prusa_bridge.cpp)
target_compile_features(prusa_bridge PRIVATE cxx_std_17)
target_link_libraries(prusa_bridge PRIVATE libslic3r)

set_target_properties(prusa_bridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${KUKA_NATIVE_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${KUKA_NATIVE_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${KUKA_NATIVE_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${KUKA_NATIVE_OUTPUT_DIR}"
)
