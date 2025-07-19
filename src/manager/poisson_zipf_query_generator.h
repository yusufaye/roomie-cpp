#ifndef POISSON_ZIPF_QG_H
#define POISSON_ZIPF_QG_H

#include <thread>
#include <future>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include "engine.h"
#include "utils/csv.h"
#include "networking/port.h"
#include "networking/message.h"

class PoissonZipfQueryGenerator : public Engine
{
private:
  std::string engine_name = "Query-Generator";
  double duration_;
  std::vector<std::string> domain_;
  std::string path_;
  int qps_;
  std::map<std::string, int> counter_;

public:
  void configure(const json config)
  {
    Engine::configure(config);
    auto params = config_["parameters"];
    duration_ = 60.0 * params["duration"].get<float>();
    domain_ = params["domain"];
    path_ = params["path"];
    qps_ = params["qps"].get<int>();
  }

  void run()
  {
    spdlog::debug("--- Poisson - Zipf Query Generator ---\n\tduration: {}, path: {}sec, qps: {}", duration_, qps_, path_);
    std::map<std::string, std::string> regis_data;
    for (const auto &name : domain_)
    {
      regis_data[name] = name;
    }
    Message msg("REGISTER", regis_data);
    outgoing_[0]->push(msg);

    std::string names = "";
    for (const auto name: domain_) {
      names += name + ", ";
    }
    spdlog::debug("Registering model variants: {}", names);

    auto data = loadTrace(path_);
    const int N = domain_.size();

    for (auto [idx, timestamps] : data)
    {
      std::string model = domain_[idx % N];
      forward_query_threads_.emplace_back([this, model, timestamps]()
                                          { sendQueries(model, timestamps); });
    }

    // DEBUG
    forward_query_threads_.emplace_back([this]()
                                        { debug(); });
    // DEBUG

    auto start = std::chrono::steady_clock::now();
    while (std::any_of(forward_query_threads_.begin(), forward_query_threads_.end(), [](std::thread &t)
                       { return t.joinable(); }))
    {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      // spdlog::debug( "Progress: " << std::min(elapsed, duration_) << " / " << duration_ << " seconds\r";
    }

    Message finished_msg("FINISHED");
    outgoing_[0]->push(finished_msg);
  }

private:
  std::vector<std::thread> forward_query_threads_;

  std::unordered_map<int, std::vector<double>> loadTrace(const std::string &filepath)
  {
    std::unordered_map<int, std::vector<double>> data;
    try
    {
      io::CSVReader<2> in(filepath);
      in.read_header(io::ignore_extra_column, "timestamp", "model");
      float timestamp;
      int idx;
      while (in.read_row(timestamp, idx))
      {
        if (timestamp <= duration_)
        {
          data[idx].push_back(timestamp);
        }
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️ Error during trace loading: " << e.what() << '\n';
    }

    return data;
  }

  void sendQueries(const std::string &app_id, std::vector<double> timestamps)
  {
    counter_[app_id] = 0;
    std::sort(timestamps.begin(), timestamps.end());
    double time = 0;
    for (const double timestamp : timestamps)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(timestamp - time));
      Message msg(std::chrono::system_clock::now().time_since_epoch().count(), "QUERY", {{"app_id", app_id}});
      outgoing_[0]->push(msg);
      time = timestamp;
      counter_[app_id]++;
    }
  }

  void debug()
  {
    while (true)
    {
      int start_total = 0;
      int end_total = 0;
      for (const auto [_, count] : counter_)
      {
        start_total += count;
      }
      std::this_thread::sleep_for(std::chrono::duration<double>(1));
      for (const auto [_, count] : counter_)
      {
        end_total += count;
      }
      // spdlog::debug( "QPS: " + std::to_string(end_total - start_total) << std::endl;
    }
  }
};

#endif // POISSON_ZIPF_QG_H