cmake_minimum_required(VERSION 3.20)

project(cuda_telemetry_engine
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(telemetry_cpu
    src/validation.cpp
    src/generator.cpp
)

target_include_directories(telemetry_cpu
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(telemetry_cpu PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(telemetry_engine
    src/main.cpp
)

target_link_libraries(telemetry_engine
    PRIVATE telemetry_cpu
)

# CTest supplies BUILD_TESTING and registers tests for the ctest command.
option(TELEMETRY_ENABLE_CUDA "Build the CUDA backend" OFF)
include(CTest)
if(BUILD_TESTING)
    add_executable(telemetry_tests tests/validation_tests.cpp)
    target_link_libraries(telemetry_tests PRIVATE telemetry_cpu)
    target_compile_options(telemetry_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME cpu_validation COMMAND telemetry_tests)

    add_executable(generator_tests tests/generator_tests.cpp)
    target_link_libraries(generator_tests PRIVATE telemetry_cpu)
    target_compile_options(generator_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME telemetry_generation COMMAND generator_tests)

    add_test(NAME cli_backend COMMAND ${CMAKE_COMMAND}
        -DENGINE=$<TARGET_FILE:telemetry_engine>
        -DCUDA_ENABLED=${TELEMETRY_ENABLE_CUDA}
        -P ${PROJECT_SOURCE_DIR}/tests/cli_tests.cmake
    )
endif()

if(TELEMETRY_ENABLE_CUDA)
    enable_language(CUDA)

    add_library(telemetry_cuda src/validation_cuda.cu)

    target_include_directories(telemetry_cuda
        PUBLIC ${PROJECT_SOURCE_DIR}/include
    )

    set_target_properties(telemetry_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )

    target_link_libraries(telemetry_engine PRIVATE telemetry_cuda)
    target_compile_definitions(telemetry_engine PRIVATE TELEMETRY_ENABLE_CUDA=1)

    if(BUILD_TESTING)
        add_executable(telemetry_cuda_tests
            tests/validation_cuda_tests.cpp
        )

        target_link_libraries(telemetry_cuda_tests
            PRIVATE telemetry_cuda telemetry_cpu
        )

        add_test(NAME cuda_validation COMMAND telemetry_cuda_tests)
    endif()
endif()
