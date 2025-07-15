# Find nlohmann/json and zaphoyd/websocketpp
find_package(nlohmann_json REQUIRED)
find_package(websocketpp REQUIRED)

# Create library
add_library(networking port.h message.h)
target_link_libraries(networking ${nlohmann_json_LIBRARIES} ${websocketpp_LIBRARIES})
target_include_directories(networking PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(networking PROPERTIES LINKER_LANGUAGE CXX)