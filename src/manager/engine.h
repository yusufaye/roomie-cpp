#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include "utils/general.h"
#include "networking/port.h"
#include "networking/message.h"

namespace fs = std::filesystem;

class Engine
{
protected:
  int id_ = 0;
  bool running_ = true;
  std::string engine_name_ = "Engine";
  json config_;
  std::string log_directory_;
  std::vector<InPort *> incoming_;
  std::vector<OutPort *> outgoing_;
  RandomGenerator generator_;

public:
  virtual void configure(const std::string &path)
  {
    std::cout << "⚠️[" << engine_name_ << "] About to load configuration from " << "'" << path << "'" << std::endl;
    try
    {
      // read a JSON file
      std::ifstream i(path);
      i >> config_;
      i.close();
    }
    catch(const std::exception& e)
    {
      std::cerr << "⛔️Error loading configuration" << "\n\t" << e.what() << '\n';
      return;
    }
    id_ = config_["id"].get<int>();
    // std::cout << "[" << engine_name_ << "] Loaded configuration: " << config_ << std::endl;
    if (config_["parameters"].contains("log_dir"))
    {
      log_directory_ = config_["parameters"]["log_dir"];
    }

    if (config_.contains("host") && config_.contains("port") && config_["port"].get<int>() > 0)
    {
      auto in = new InPort(config_["host"], config_["port"],
                           [this](const Message &msg)
                           { push(msg); });
      incoming_.push_back(in);
    }
    for (auto &remote : config_["remote_engines"])
    {
      int worker_id = generator_.next();
      remote["id"] = worker_id;
      auto out = new OutPort(worker_id, remote["remote_host"], remote["remote_port"]);
      outgoing_.push_back(out);
    }
  }

  virtual void push(const Message &msg)
  {
    throw std::runtime_error("Not implemented");
  }

  virtual void run()
  {
    throw std::runtime_error("Not implemented");
  }

  void start()
  {
    try
    {
      run(); // This could be a blocking loop
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️[ENGINE]-run: " << e.what() << std::endl;
      fs::remove(log_directory_);
      throw e;
    }

    std::cout << "--- " + engine_name_ + " terminated! ---" << std::endl;
  }

  std::vector<OutPort *> get_outgoing()
  {
    return outgoing_;
  }

  std::vector<InPort *> get_incoming()
  {
    return incoming_;
  }

  RandomGenerator *get_generator()
  {
    return &generator_;
  }

  json get_config() const { return config_; }
};

#endif // ENGINE_H