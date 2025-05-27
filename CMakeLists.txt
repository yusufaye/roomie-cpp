cmake_minimum_required(VERSION 3.10)
project(Roomie)

# Find LibTorch
find_package(Torch REQUIRED HINTS /home/yusufaye/roomie-cpp/package/libtorch)

# Add src directory
add_subdirectory(src)
