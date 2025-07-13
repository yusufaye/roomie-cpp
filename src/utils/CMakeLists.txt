include(FetchContent)

# Fetch csv-parser and make it available to my-executable
FetchContent_Declare(
  csv  
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE 
  GIT_TAG 2.2.1
) 

FetchContent_MakeAvailable(csv)

# Create library
add_library(utils profiler.h kernels.h datastore.h general.h constants.h queue.h load_balancing.h)
target_link_libraries(utils csv)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})