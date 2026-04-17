cmake_minimum_required(VERSION 3.18)
project(tiny-llm
    VERSION 2.0.0
    DESCRIPTION "Lightweight LLM inference engine with W8A16 quantization"
    LANGUAGES CXX
)

include(CheckLanguage)
include(FetchContent)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA toolkit with nvcc is required to build tiny-llm. Set CUDAToolkit_ROOT or ensure nvcc is available on PATH.")
endif()
enable_language(CUDA)

# ── C++ / CUDA 标准 ──────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── 外部依赖 ─────────────────────────────────────────────────────

# spdlog (logging library)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

# ── CUDA 配置 ────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
endif()

# ── 构建类型 ─────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── 主库 ─────────────────────────────────────────────────────────
file(GLOB_RECURSE SOURCES "src/*.cpp" "kernels/*.cu")
# Exclude main.cpp from library
list(FILTER SOURCES EXCLUDE REGEX "src/main\\.cpp$")

add_library(tiny_llm STATIC ${SOURCES})
add_library(tiny_llm::tiny_llm ALIAS tiny_llm)

target_include_directories(tiny_llm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(tiny_llm PUBLIC CUDA::cudart spdlog::spdlog)

target_compile_options(tiny_llm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

# ── 主程序 ───────────────────────────────────────────────────────
add_executable(tiny_llm_demo src/main.cpp)
target_link_libraries(tiny_llm_demo PRIVATE tiny_llm)

# ── 测试 ─────────────────────────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # RapidCheck (property-based testing) with GTest support
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG master
    )
    set(RAPIDCHECK_ENABLE_GTEST ON CACHE BOOL "Enable GTest support for rapidcheck" FORCE)
    FetchContent_MakeAvailable(rapidcheck)

    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.cu")
    add_executable(tiny_llm_tests ${TEST_SOURCES})
    target_link_libraries(tiny_llm_tests PRIVATE
        tiny_llm
        GTest::gtest_main
        rapidcheck
        rapidcheck_gtest
        CUDA::cudart
    )
    target_include_directories(tiny_llm_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${rapidcheck_SOURCE_DIR}/include
        ${rapidcheck_SOURCE_DIR}/extras/gtest/include
    )
    target_compile_options(tiny_llm_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    include(GoogleTest)
    gtest_discover_tests(tiny_llm_tests)
endif()
