#ifndef PORT_H
#define PORT_H

#include <string>
#include <queue>
#include <memory>
#include <thread>
#include <websocketpp/server.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include "utils/queue.h"
#include "message.h"

typedef websocketpp::client<websocketpp::config::asio> client;
typedef websocketpp::server<websocketpp::config::asio> server;

using namespace std;

class InPort
{
public:
  InPort(const std::string &host, int port, std::function<void(const Message &)> callback)
      : host_(host), port_(port), callback_(callback), server_()
  {
    try
    {
      std::cout << "[InPort] Host: " + host + ", Port: " + std::to_string(port) << std::endl;
      // Set logging settings
      server_.get_alog().set_channels(websocketpp::log::alevel::none);

      // Initialize Asio
      server_.init_asio();

      // Handle WebSocket connections
      server_.set_open_handler([this](websocketpp::connection_hdl hdl)
                               { std::cout << "[InPort] Client connected." << std::endl; });

      server_.set_close_handler([this](websocketpp::connection_hdl hdl)
                                { std::cout << "👋🏻[InPort] Server disconnected." << std::endl; });
      server_.set_fail_handler([this](websocketpp::connection_hdl)
                               {
            connected_ = false;
            std::cerr << "⛔️[InPort] Connection failed to host " + host_ + " and port " + std::to_string(port_) << std::endl; });
      server_.set_message_handler([this](websocketpp::connection_hdl hdl, server::message_ptr msg)
                                  { message_queue_.push(msg->get_payload()); });

      server_.set_open_handler([this](websocketpp::connection_hdl)
                               { connected_ = true; });

      server_.listen(port);
      // Start the server accept loop
      server_.start_accept();

      server_thread_ = std::thread([this]()
                                   { server_.run(); });

      consumer_thread_ = std::thread(&InPort::run, this);
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️[InPort] Error while trying to set input connection at host: " + host_ +
                       " and port: " + std::to_string(port_) + "\n\t"
                << e.what() << '\n';
    }
  }

  ~InPort()
  {
    if (consumer_thread_.joinable())
      consumer_thread_.join();
    if (server_thread_.joinable())
      server_thread_.join();
    server_.stop();
  }

  std::string to_string() const
  {
    return "InPort('host': " + host_ +
           ", 'port': " + std::to_string(port_) +
           ", 'qsize': " + std::to_string(message_queue_.size()) + ")";
  }

  std::string get_host() const { return host_; }
  int get_port() const { return port_; }

private:
  void run()
  {
    std::string data;
    while (true)
    {
      data = message_queue_.pop();
      Message message;
      message.deserialize(data);
      callback_(message);
      if (message.getType() == "FINISHED")
      {
        break;
      }
    }
  }

  server server_;
  std::string host_;
  int port_;
  bool connected_;
  std::function<void(Message)> callback_;
  BlockingQueue<std::string> message_queue_;
  std::thread server_thread_;
  std::thread consumer_thread_;
};

class OutPort
{
public:
  OutPort(int id, const std::string &remote_host, int remote_port)
      : id_(id), remote_host_(remote_host), remote_port_(remote_port),
        client_()
  {
    std::cout << "[OutPort] Host: " + remote_host + ", Port: " + std::to_string(remote_port) << std::endl;

    // Set logging to be pretty verbose (everything except message payloads)
    client_.get_alog().set_channels(websocketpp::log::alevel::none);

    // ex. "ws://localhost:9002"
    url_ = "ws://" + remote_host + ":" + std::to_string(remote_port);

    // Initialize ASIO
    client_.init_asio();
    client_.set_open_handler([this](websocketpp::connection_hdl hdl)
                             {
        this->hdl_ = hdl;
        connected_ = true;
        runner_thread_ = std::thread(&OutPort::run, this);
        std::cout << "✅[OutPort] Connected successfully!" << std::endl; });
    // Handle WebSocket connections

    client_.set_close_handler([&](websocketpp::connection_hdl hdl)
                              { std::cout << "👋🏻[OutPort] Client disconnected." << std::endl; });

    client_.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg)
                                {
                                  Message message(msg->get_payload());
                                  std::cout << "[OutPort] Received message" << message.to_string() << std::endl;
                                  // Handle message
                                });
    client_.set_fail_handler([this](websocketpp::connection_hdl)
                             {
            connected_ = false;
            std::cerr << "⛔️[OutPort] Connection failed to host " + remote_host_ + " and port " + std::to_string(remote_port_) + ". Retrying...\n";
            schedule_retry(); });

    connect();

    // Start the ASIO io_service run loop
    // this will cause a single connection to be made to the server.
    client_thread_ = std::thread([this]()
                                 { client_.run(); });
  }

  ~OutPort()
  {
    if (runner_thread_.joinable())
      runner_thread_.join();
    if (client_thread_.joinable())
      client_thread_.join();
    Message message("FINISHED");
    message_queue_.push(message); // Empty string signals shutdown
    client_.stop();
  }

  void connect()
  {
    websocketpp::lib::error_code ec;
    client::connection_ptr con = client_.get_connection(url_, ec);
    if (ec)
    {
      std::cerr << "Could not create connection: " << ec.message() << "\n";
      return;
    }
    client_.connect(con);
  }

  void schedule_retry()
  {
    if (retry_count_ >= max_retries_)
    {
      std::cerr << "⛔️ Max retries reached. Giving up.\n";
      return;
    }
    retry_count_++;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    connect();
  }

  void push(const Message &msg)
  {
    message_queue_.push(msg);
  }

  std::string getRemoteHost()
  {
    return remote_host_;
  }
  int getRemotePort()
  {
    return remote_port_;
  }
  int getId()
  {
    return id_;
  }

  std::string to_string() const
  {
    return "OutPort('remote host': " + remote_host_ +
           ", 'remote port': " + std::to_string(remote_port_) +
           ", 'qsize': " + std::to_string(message_queue_.size()) + ")";
  }

private:
  void run()
  {
    while (!connected_)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    try
    {
      while (true)
      {
        Message message = message_queue_.pop();
        client_.send(hdl_, message.serialize(), websocketpp::frame::opcode::text);
        if (message.getType() == "FINISHED")
        {
          cout << "--- Will close connection---" << endl;
          break;
        }
      }
    }
    catch(const std::exception& e)
    {
      std::cerr << "⛔️ Connection lost\n\t" << e.what() << '\n';
    }
  }

  std::string remote_host_;
  int remote_port_;
  int id_;
  std::string url_;
  client client_;
  bool connected_;
  BlockingQueue<Message> message_queue_; // Not thread-safe!  std::mutex queue_mutex_;
  websocketpp::connection_hdl hdl_;
  std::thread client_thread_;
  std::thread runner_thread_;
  int retry_count_;
  const int max_retries_ = 20;
};

#endif // PORT_H