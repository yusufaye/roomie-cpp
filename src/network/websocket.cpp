// client.cpp
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/server.hpp>
#include "websocket.h"

typedef websocketpp::client<websocketpp::config::asio> client;
typedef websocketpp::server<websocketpp::config::asio> server;
typedef websocketpp::config::asio_client::message_type::ptr message_ptr_c;

WebSocketClient::WebSocketClient(const std::string url) : client_(), url_(url)
{
  // "ws://localhost:9002"
  try
  {
    // Set logging to be pretty verbose (everything except message payloads)
    client_.set_access_channels(websocketpp::log::alevel::all);
    client_.clear_access_channels(websocketpp::log::alevel::frame_payload);

    // Initialize ASIO
    client_.init_asio();
    client_.set_open_handler([this](websocketpp::connection_hdl hdl)
                             {
                    
        hdl_ = hdl;
        connected_ = true;
        std::cout << "Connected successfully!" << std::endl; });
    client_.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg)
                                {
                                  Message message(msg->get_payload());
                                  std::cout << "[Client] Received message: < " << message.toString() << " >" << std::endl;
                                  // Handle message
                                });

    websocketpp::lib::error_code ec;
    client::connection_ptr con = client_.get_connection(url, ec);
    if (ec)
    {
      std::cout << "could not create connection because: " << ec.message() << std::endl;
      exit(1);
    }

    // Note that connect here only requests a connection. No network messages are
    // exchanged until the event loop starts running in the next line.
    client_.connect(con);

    // Start the ASIO io_service run loop
    // this will cause a single connection to be made to the server. client_.run()
  }
  catch (websocketpp::exception const &e)
  {
    std::cout << e.what() << std::endl;
  }
}

void WebSocketClient::send(Message &message)
{
  std::string payload = message.toString();
  websocketpp::frame::opcode::value op = websocketpp::frame::opcode::text;
  websocketpp::lib::error_code ec;
  client_.send(hdl_, payload, op, ec);
  if (ec)
  {
    std::cerr << "Error sending message: " << ec.message() << std::endl;
  }
}

void WebSocketClient::run()
{
  // will exit when this connection is closed.
  client_.run();
}

bool WebSocketClient::is_connected() {
  return connected_;
}

WebSocketServer::WebSocketServer(const std::string url) : server_(), url_(url)
{
  // Set logging settings
  server_.set_access_channels(websocketpp::log::alevel::all);
  server_.clear_access_channels(websocketpp::log::alevel::frame_payload);

  // Initialize Asio
  server_.init_asio();
  server_.set_message_handler([this](websocketpp::connection_hdl hdl, server::message_ptr msg)
                              {
                                Message message(msg->get_payload());
                                std::cout << "[Server] Received message: < " << message.toString() << " >" << std::endl;
                                // Handle message
                              });

  // Listen on port 9002
  server_.listen(9002);

  // Start the server accept loop
  server_.start_accept();
}

void WebSocketServer::run()
{
  server_.run();
}