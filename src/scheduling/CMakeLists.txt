# Create library

add_library(scheduling base_scheduler.h infaas_scheduler.h infaas_v1_scheduler.h usher_scheduler.h roomie_scheduler.h roomie_v1_scheduler.h roomie_v2_scheduler.h roomie_v3_scheduler.h)
target_link_libraries(scheduling)
target_include_directories(scheduling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(scheduling PROPERTIES LINKER_LANGUAGE CXX)