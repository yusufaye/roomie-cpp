# Add models directory
add_subdirectory(models)

# Add network directory
add_subdirectory(network)

# Add utils directory
add_subdirectory(utils)

# Create executable
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} models network utils)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})