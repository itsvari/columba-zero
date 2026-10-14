cmake_minimum_required(VERSION 3.26)
project(columba_native VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    find_path(OpenMP_CXX_INCLUDE_DIR omp.h PATHS /opt/homebrew/opt/libomp/include)
    find_library(OpenMP_omp_LIBRARY omp PATHS /opt/homebrew/opt/libomp/lib)
    if(OpenMP_omp_LIBRARY AND OpenMP_CXX_INCLUDE_DIR)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES omp)
    endif()
endif()

find_package(ncnn CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_directory
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_directory}" NO_DEFAULT_PATH)

add_library(columba_detection STATIC src/detection.cpp src/detector.cpp)
target_include_directories(columba_detection PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(columba_detection PUBLIC ncnn ${OpenCV_LIBS})
target_compile_options(columba_detection PRIVATE -Wall -Wextra -Wpedantic)

nanobind_add_module(columba_native src/bindings.cpp)
target_link_libraries(columba_native PRIVATE columba_detection)
target_compile_options(columba_native PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS columba_native LIBRARY DESTINATION .)
install(FILES columba_native.pyi DESTINATION .)

include(CTest)
if(BUILD_TESTING)
    add_executable(columba_detection_tests tests/detection_tests.cpp)
    target_link_libraries(columba_detection_tests PRIVATE columba_detection)
    add_test(NAME detection COMMAND columba_detection_tests)
endif()
