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
      spdlog::debug("[InPort] Host: {}, Port: {}" + host, port);
      // Set logging settings
      server_.get_alog().set_channels(websocketpp::log::alevel::none);

      // Initialize Asio
      server_.init_asio();

      // Handle WebSocket connections
      server_.set_open_handler([this](websocketpp::connection_hdl hdl)
                               { spdlog::debug("[InPort] Client connected."); });

      server_.set_close_handler([this](websocketpp::connection_hdl hdl)
                                { spdlog::debug("👋🏻[InPort] Server disconnected."); });
      server_.set_fail_handler([this](websocketpp::connection_hdl)
                               {
            connected_ = false;
            spdlog::error("⛔️[InPort] Connection failed to host {} and port {}" ,host_ ,port_); });
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
      spdlog::error("⛔️[InPort] Error while trying to set input connection at host: {}, and port: {}", host_, port_, e.what());
    }
  }

  ~InPort()
  {
    server_.stop();
    message_queue_.push(""); // Empty string signals shutdown
    if (consumer_thread_.joinable())
    {
      consumer_thread_.join();
    }
    if (server_thread_.joinable())
    {
      server_thread_.join();
    }
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
      if (data.empty())
      {
        break;
      }
      Message message;
      message.deserialize(data);
      callback_(message);
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
    spdlog::debug("[OutPort] Host: {}, Port: {}", remote_host, remote_port);

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
        spdlog::debug("✅[OutPort] Connected successfully!"); });
    // Handle WebSocket connections

    client_.set_close_handler([&](websocketpp::connection_hdl hdl)
                              { spdlog::debug("👋🏻[OutPort] Client disconnected."); });

    client_.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg)
                                {
                                  Message message(msg->get_payload());
                                  spdlog::debug("[OutPort] Received message: {}", message.to_string());
                                  // Handle message
                                });
    client_.set_fail_handler([this](websocketpp::connection_hdl)
                             {
            connected_ = false;
            spdlog::error("⛔️[OutPort] Connection failed to host {} and port {}\n\tRetrying...", remote_host_ ,remote_port_);
            schedule_retry(); });

    connect();

    // Start the ASIO io_service run loop
    // this will cause a single connection to be made to the server.
    client_thread_ = std::thread([this]()
                                 { client_.run(); });
  }

  ~OutPort()
  {
    Message message("FINISHED");
    message_queue_.push(message); // Empty string signals shutdown
    client_.stop();
    if (runner_thread_.joinable())
    {
      runner_thread_.join();
    }
    if (client_thread_.joinable())
    {
      client_thread_.join();
    }
  }

  void connect()
  {
    websocketpp::lib::error_code ec;
    client::connection_ptr con = client_.get_connection(url_, ec);
    if (ec)
    {
      spdlog::error("Could not create connection: {}", ec.message());
      return;
    }
    client_.connect(con);
  }

  void schedule_retry()
  {
    if (retry_count_ >= max_retries_)
    {
      spdlog::error("⛔️ Max retries reached. Giving up.");
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
    catch (const std::exception &e)
    {
      spdlog::error("⛔️ Connection lost\n\t{}", e.what());
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