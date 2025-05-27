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
  WebSocketClient() : client_()
  {
    client_.init_asio();
    client_.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg)
                                {
                                  Message message(msg->get_payload());
                                  std::cout << "Received message: " << message.toString() << std::endl;
                                  // Handle message
                                });
  }

  // void connect(const std::string &uri)
  // {
  //   client_.connect(uri);
  // }

  // void send_message(const Message &message)
  // {
  //   client_.send(message.toString());
  // }

  void run()
  {
    client_.run();
  }

private:
  client client_;
};


class WebSocketServer
{
public:
  WebSocketServer() : server_()
  {
    server_.init_asio();
    server_.set_reuse_addr(true);
    server_.listen(9002);
    server_.start_accept();
  }

  void run()
  {
    server_.run();
  }

  void on_message(websocketpp::connection_hdl hdl, server::message_ptr msg)
  {
    Message message(msg->get_payload());
    std::cout << "Received message: " << message.toString() << std::endl;
    // Handle message
  }

private:
  server server_;
};