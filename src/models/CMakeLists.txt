# Create library
add_library(models model.cpp model.hpp)
target_link_libraries(models ${TORCH_LIBRARIES})
target_include_directories(models PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})