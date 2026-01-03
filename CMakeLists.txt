cmake_minimum_required(VERSION 3.18)
project(tiny-llm LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/kernels)

# Source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.cu"
    "kernels/*.cu"
)

# Main library
add_library(tiny_llm STATIC ${SOURCES})
target_link_libraries(tiny_llm CUDA::cudart)

# Enable testing
enable_testing()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Fetch RapidCheck for property-based testing
FetchContent_Declare(
    rapidcheck
    GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
    GIT_TAG master
)
FetchContent_MakeAvailable(rapidcheck)

# Test executable
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.cu")
add_executable(tiny_llm_tests ${TEST_SOURCES})
target_include_directories(tiny_llm_tests PRIVATE 
    ${rapidcheck_SOURCE_DIR}/extras/gtest/include
)
target_link_libraries(tiny_llm_tests 
    tiny_llm 
    GTest::gtest_main 
    rapidcheck
    CUDA::cudart
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(tiny_llm_tests)

# Main executable (demo)
add_executable(tiny_llm_demo src/main.cpp)
target_link_libraries(tiny_llm_demo tiny_llm CUDA::cudart)
