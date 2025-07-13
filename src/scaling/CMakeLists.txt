# Create library

add_library(scaling auto_scaler.h)
target_link_libraries(scaling)
target_include_directories(scaling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(scaling PROPERTIES LINKER_LANGUAGE CXX)