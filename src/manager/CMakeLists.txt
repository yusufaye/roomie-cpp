include(FetchContent)

# Fetch csv-parser and make it available to my-executable
FetchContent_Declare(
  csv  
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE 
  GIT_TAG 2.2.1
) 

FetchContent_MakeAvailable(csv)

# Find nlohmann/json and zaphoyd/websocketpp
find_package(nlohmann_json REQUIRED)

# Create library
add_library(manager engine.h controller.h poisson_zipf_query_generator.h worker.h)
target_link_libraries(manager ${nlohmann_json_LIBRARIES} csv)
target_include_directories(manager PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})