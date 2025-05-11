cmake_minimum_required(VERSION 3.10)
project(nerf_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${nlohmann_json_INCLUDE_DIRS}
)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
)

# Create executable
add_executable(nerf_train examples/train.cpp ${SOURCES})

# Link libraries
target_link_libraries(nerf_train
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

# Set compiler flags
if(MSVC)
    target_compile_options(nerf_train PRIVATE /W4)
else()
    target_compile_options(nerf_train PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Set CUDA properties if available
if(TORCH_CUDA_AVAILABLE)
    set_target_properties(nerf_train PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif() 