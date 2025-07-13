# Add models directory
add_subdirectory(models)

# Add networking directory
add_subdirectory(networking)

# Add utils directory
add_subdirectory(utils)

# Add scheduling directory
add_subdirectory(scheduling)

# Add manager directory
add_subdirectory(manager)

# Create executable
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} models networking utils scheduling manager)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})