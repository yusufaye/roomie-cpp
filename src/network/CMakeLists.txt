# Find nlohmann/json and zaphoyd/websocketpp
find_package(nlohmann_json REQUIRED)
find_package(websocketpp REQUIRED)

# Create library
add_library(network websocket.cpp message.h)
target_link_libraries(network ${nlohmann_json_LIBRARIES} ${websocketpp_LIBRARIES})
target_include_directories(network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})