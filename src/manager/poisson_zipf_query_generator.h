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
#include "csv.hpp"
#include "engine.h"
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

public:
  void configure(const std::string &path)
  {
    Engine::configure(path);
    auto params = config_["parameters"];
    duration_ = 60.0 * params["duration"].get<float>();
    domain_ = params["domain"];
    path_ = params["path"];
    qps_ = params["qps"].get<int>();
  }

  void run()
  {
    std::cout << "--- Poisson - Zipf Query Generator ---\n"
              << "\tduration: " << duration_ << "sec, qps: " << qps_
              << ", path: " << path_ << std::endl;
    std::map<std::string, std::string> regis_data;
    for (const auto &name : domain_)
    {
      regis_data[name] = name;
    }
    Message msg("REGISTER", regis_data);
    outgoing_[0]->push(msg);

    std::cout << "Registering model variants: ";
    for (const auto &name : domain_)
      std::cout << name << ", ";
    std::cout << std::endl;

    auto data = loadTrace(path_);
    const int N = domain_.size();

    for (auto [idx, timestamps] : data)
    {
      std::string model = domain_[idx % N];
      forward_query_threads_.emplace_back([this, model, timestamps]()
                                          { sendQueries(model, timestamps); });
    }

    auto start = std::chrono::steady_clock::now();
    while (std::any_of(forward_query_threads_.begin(), forward_query_threads_.end(), [](std::thread &t)
                       { return t.joinable(); }))
    {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      std::cout << "Progress: " << std::min(elapsed, duration_) << " / " << duration_ << " seconds\r";
    }
    std::cout << "\n";

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
      float timestamp;
      int idx;
      csv::CSVReader reader(filepath);
      for (csv::CSVRow &row : reader)
      {
        // model,timestamp
        timestamp = row["timestamp"].get<float>();
        idx = row["model"].get<int>();
        if (timestamp <= duration_)
        {
          data[idx].push_back(timestamp);
        }
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️Error during trace loading: " << e.what() << '\n';
    }

    return data;
  }

  void sendQueries(const std::string &model, std::vector<double> timestamps)
  {
    std::sort(timestamps.begin(), timestamps.end());
    double time = 0;
    for (const double timestamp : timestamps)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(timestamp - time));
      Message msg(std::chrono::system_clock::now().time_since_epoch().count(), "QUERY", {{"app_id", model}});
      outgoing_[0]->push(msg);
      time = timestamp;
    }
  }
};

#endif // POISSON_ZIPF_QG_H