option(ENABLE_CUDA "Enable CUDA support" ON)

# Find CUDA
find_package(CUDA)

# Find nlohmann/json and zaphoyd/websocketpp
find_package(nlohmann_json REQUIRED)

# Check if build for CUDA available env.
if(CUDA_FOUND OR ENABLE_CUDA)
  # Find libtorch
  find_package(Torch REQUIRED HINTS /usr/local/libtorch)
  # find_package(Torch REQUIRED HINTS /usr/local/libtorch/libtorch)

  add_library(manager engine.h worker.h)
  target_link_libraries(manager ${nlohmann_json_LIBRARIES} ${TORCH_LIBRARIES})
else()
  add_library(manager engine.h controller.h poisson_zipf_query_generator.h)
  target_link_libraries(manager ${nlohmann_json_LIBRARIES})
endif()

target_include_directories(manager PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(manager PROPERTIES LINKER_LANGUAGE CXX)
