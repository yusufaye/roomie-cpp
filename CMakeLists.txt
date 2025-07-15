cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE ARCH
)

if(${ARCH} MATCHES "aarch64")
  set(ARCH_FLAG "-march=armv8-a")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
elseif(${ARCH} MATCHES "x86_64")
  set(CMAKE_CXX_COMPILER "g++")
  set(CMAKE_CUDA_ARCHITECTURES 80)

else()
  message(FATAL_ERROR "Unsupported architecture: ${ARCH}")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-lstdc++fs)
endif()

project(Roomie)

# Find NumCpp
find_package(Threads REQUIRED)


# Add src directory
add_subdirectory(src)
