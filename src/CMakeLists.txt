option(ENABLE_CUDA "Enable CUDA support" ON)

# Find CUDA
find_package(CUDA)

# Set the executable name based on CUDA availability§/L.
if(CUDA_FOUND OR ENABLE_CUDA)
  find_package(Torch REQUIRED HINTS /home/yusufaye/roomie-cpp/package/libtorch)

  set(EXECUTABLE_NAME "${PROJECT_NAME}_CUDA")
  add_definitions(-DCUDA_AVAILABLE)
else()
  set(EXECUTABLE_NAME "${PROJECT_NAME}")
endif()

# Add networking directory
add_subdirectory(networking)

# Add utils directory
add_subdirectory(utils)

# Add scheduling directory
add_subdirectory(scheduling)

# Add manager directory
add_subdirectory(manager)

# Create executable
add_executable(${EXECUTABLE_NAME} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} networking utils scheduling manager Threads::Threads) # if pthread is not included by default.
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})