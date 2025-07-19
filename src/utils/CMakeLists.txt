# Create library
add_library(utils profiler.h kernels.h datastore.h general.h constants.h queue.h load_balancing.h csv.h csv_writer.h)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)