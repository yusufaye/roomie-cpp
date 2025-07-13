# Find nlohmann/json and zaphoyd/websocketpp
find_package(nlohmann_json REQUIRED)
find_package(websocketpp REQUIRED)

# Create library
add_library(networking INTERFACE port.h message.h)
target_link_libraries(networking ${nlohmann_json_LIBRARIES} ${websocketpp_LIBRARIES})
target_include_directories(networking INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})