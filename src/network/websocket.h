// client.cpp
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/server.hpp>
#include "message.h"

typedef websocketpp::client<websocketpp::config::asio> client;
typedef websocketpp::server<websocketpp::config::asio> server;

class WebSocketClient
{
public:
  WebSocketClient(const std::string url);
  void send(Message &message);
  void run();
  bool is_connected();
private:
  client client_;
  std::string url_;
  bool connected_;
  websocketpp::connection_hdl hdl_;
};


class WebSocketServer
{
public:
  WebSocketServer(const std::string url);
  void run();

private:
  server server_;
  std::string url_;
};