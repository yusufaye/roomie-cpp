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

using namespace std::chrono;

class PoissonZipfQueryGenerator : public Engine
{
private:
  std::string engine_name = "Query-Generator";
  std::map<std::string, int64_t> counter_;
  std::vector<std::string> domain_;
  int scale_;
  std::string path_;
  double duration_;

public:
  void configure(const json config)
  {
    Engine::configure(config);
    auto params = config_["parameters"];
    duration_ = 60.0 * params["duration"].get<float>();
    domain_ = params["domain"];
    scale_ = params["scale"];
    path_ = params["path"];
  }

  void run()
  {
    spdlog::info("--- Poisson - Zipf Query Generator ---\n\t* path: {}\n\t* scale: {}\n\t* duration: {}", path_, scale_, duration_);
    std::map<std::string, std::string> regis_data;
    for (const auto &name : domain_)
    {
      regis_data[name] = name;
    }
    Message msg("REGISTER", regis_data);
    outgoing_[0]->push(msg);

    std::string names = "";
    for (const auto name : domain_)
    {
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
    // forward_query_threads_.emplace_back([this]()
    //                                     { debug(); });
    // DEBUG

    auto start = std::chrono::steady_clock::now();
    while (std::any_of(forward_query_threads_.begin(), forward_query_threads_.end(), [](std::thread &t)
                       { return t.joinable(); }))
    {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      auto now = std::chrono::steady_clock::now();
      double delay = std::chrono::duration<double>(now - start).count();
      spdlog::debug("Progress: {} / {} seconds", std::min(delay, duration_), duration_);
      if (delay > duration_)
      {
        break;
      }
    }
    exit(0);
  }

private:
  std::vector<std::thread> forward_query_threads_;

  std::unordered_map<int, std::vector<std::pair<double, double>>> loadTrace(const std::string &filepath)
  {
    std::unordered_map<int, std::vector<std::pair<double, double>>> data;
    try
    {
      io::CSVReader<3> in(WORKDIR + filepath);
      in.read_header(io::ignore_extra_column, "delay", "timestamp", "model");
      double timestamp;
      double delay;
      int idx;
      while (in.read_row(delay, timestamp, idx))
      {
        if (timestamp <= duration_)
        {
          data[idx].push_back({delay, timestamp});
        }
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️ Error during trace loading: " << e.what() << '\n';
    }

    return data;
  }

  void sendQueries(const std::string &app_id, std::vector<std::pair<double, double>> series)
  {
    for (const auto [delay, timestamp] : series)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(delay));
      auto seconds = duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
      Message msg(seconds, "QUERY", {{"app_id", app_id}});
      for (int i = 0; i < scale_; i++)
      {
        outgoing_[0]->push(msg);
      }
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
      spdlog::info( "QPS: {}", end_total - start_total );
    }
  }
};

#endif // POISSON_ZIPF_QG_H