#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <optional>
#include <condition_variable>
#include "engine.h"
#include "utils/general.h"
#include "utils/datastore.h"
#include "utils/load_balancing.h"
#include "networking/port.h"
#include "networking/message.h"
#include "scaling/auto_scaler.h"
#include "scheduling/base_scheduler.h"
// #include "scheduling/usher_scheduler.h"
#include "scheduling/infaas_scheduler.h"
// #include "scheduling/roomie_scheduler.h"

enum class Approach
{
  ROOMIE,
  INFAAS,
  USHER
};

class Controller : public Engine
{
public:
  Controller(Approach approach) : approach_(approach)
  {
    if (approach_ == Approach::INFAAS)
    {
      scheduler_ = new INFaaSScheduler();
    }
    // else if (approach_ == Approach::USHER)
    // {
    //   scheduler_ = new UsherScheduler();
    // }
    // else
    // {
    //   scheduler_ = new RoomieScheduler();
    // }
  }

  void configure(const std::string &path)
  {
    Engine::configure(path);

    incoming2_ = new InPort(get_incoming()[0]->get_host(), get_incoming()[0]->get_port() + 1, [this](Message msg)
                            { this->push(msg); });

    for (const auto &outport : outgoing_)
    {
      networking_[outport->getId()] = outport;
      Worker *worker = new Worker(outport->getId());
      datastore_.register_worker(worker);
    }
  }

  void run() override
  {
    std::cout << "RUNNING CONTROLLER..." << std::endl;

    // Send HELLO messages to all outports
    for (auto &outport : outgoing_)
    {
      Message msg("HELLO", {{"worker_id", std::to_string(outport->getId())}});
      outport->push(msg);
    }

    // Launch registration, profiling, and logging loops as background threads
    std::thread registration_thread(&Controller::registration_daemon, this);
    std::thread profiling_thread(&Controller::profiling_daemon, this);

    // Auto-scaler
    autoscaler_ = new AutoScaler(scheduler_, &datastore_, [this](const std::string &app_id, Model &variant, Worker &worker)
                                 { deploy(app_id, variant, worker); }, [this](const std::string &app_id, Model &variant, Worker &worker)
                                 { stop(app_id, variant, worker); });
    std::thread autoscaler_thread = std::thread([this]()
                                                { autoscaler_->run(); });

    // Optionally join threads or coordinate their lifecycle
    registration_thread.join();
    profiling_thread.join();
    autoscaler_thread.join();
  }

  void push(const Message &msg) override
  {
    // std::cout << "👉[CONTROLLER] Recv " + msg.to_string() << std::endl;
    if (msg.getType() == "REGISTER")
    {
      registration_queue_.push(msg);
    }
    else if (msg.getType() == "QUERY")
    {
      query_queue_[msg.get_data()["app_id"]].push(msg);
    }
    else if (msg.getType() == "PROFILE_DATA")
    {
      profiling_queue_.push(msg);
    }
    else if (msg.getType() == "HELLO")
    {
      int worker_id = std::stoi(msg.get_data()["worker_id"]);
      double total_mem = std::stod(msg.get_data()["total_mem"]);
      Worker *worker = datastore_.get_worker(worker_id);
      worker->to_string();
      worker->set_total_memory(total_mem);
      std::cout << "👉[CONTROLLER] Update for " + worker->to_string() << std::endl;
      event_.set();
    }
  }

  void registration_daemon()
  {
    try
    {
      event_.wait();
      std::cout << "👉[CONTROLLER] About to run registration daemon" << std::endl;
      while (true)
      {
        Message msg = registration_queue_.pop(); // blocks until message arrives
        std::cout << "👉[CONTROLLER] New registration " << msg.to_string() << std::endl;
        std::map<std::string, std::string> variant_names = msg.get_data(); // assumed typed extraction
        for (auto &[app_id, variant_name] : variant_names)
        {
          std::cout << "👉[CONTROLLER] About to register app " << app_id << std::endl;
          datastore_.register_app(variant_name, variant_name);
          std::vector<Worker *> workers = datastore_.get_workers();
          std::vector<std::string> names;
          for (const auto name : datastore_.get_registered(app_id))
          {
            names.push_back(name);
          }
          std::pair<Model *, Worker *> result = scheduler_->schedule(workers, names);
          // Model *variant = result.first;
          // Worker *worker = result.second;
          deploy(app_id, *result.first, *result.second);

          forward_query_threads_.emplace_back([this, variant_name]()
                                              {
                                                query_daemon(variant_name); // starts query loop per variant
                                              });
          std::cout << "👉[CONTROLLER] Registered app " << app_id << std::endl;
        }

        autoscaler_->set_event();
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️Error with registration daemon\n\t" << e.what() << '\n';
    }
  }

  void profiling_daemon()
  {
    try
    {
      while (true)
      {
        auto msg = profiling_queue_.pop(); // [TODO] Update load balancing weights.

        int worker_id = std::stoi(msg.get_data()["worker_id"]);
        json j = json::parse(msg.get_data()["variants"]);
        for (auto worker : datastore_.get_workers())
        {
          if (worker->get_id() == worker_id)
          {
            for (const auto &item : j)
            {
              for (auto variant : worker->get_variants())
              {
                if (variant->id == item["variant_id"].get<int>())
                {
                  variant->set_throughput(item["throughput"].get<float>());
                  auto input_rates = item["input_rate"].get<std::vector<int>>();
                  for (size_t i = 0; i < input_rates.size(); i++)
                  {
                    variant->input_rates[i] = input_rates[i];
                  }
                  break;
                }
              }
            }
          }
          break;
        }
        update_load_balancer();
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️Error with profiling daemon\n\t" << e.what() << '\n';
    }
  }

  void update_load_balancer()
  {
    for (const auto &[app_id, names] : datastore_.get_registration())
    {
      auto variant_workers = datastore_.get_variant_workers(app_id);
      if (variant_workers.empty())
      {
        continue;
      }

      std::vector<double> raw_weights;
      for (const auto &pair : variant_workers)
      {
        Model *variant = pair.first;
        double throughput = variant->compute_throughput();
        if (throughput == 0)
        {
          std::cerr << "Warning: Zero throughput for variant " << variant->id << std::endl;
          continue;
        }
        raw_weights.push_back(variant->compute_workload() / throughput);
      }

      std::vector<int> weights;
      double total = 0.0;
      for (const auto &w : raw_weights)
      {
        total += std::ceil(w);
      }

      for (const auto &w : raw_weights)
      {
        int adjusted = std::ceil(total - std::ceil(w)) + 1;
        weights.push_back(adjusted);
      }

      for (size_t i = 0; i < variant_workers.size(); ++i)
      {
        const std::string key = std::to_string(variant_workers[i].first->id) + "_" + std::to_string(variant_workers[i].second->get_id());
        variant_worker_map_[key] = {variant_workers[i].first, variant_workers[i].second};
        loadb_.set(app_id, key, weights[i]);
      }
    }
    // std::cout << "👉[CONTROLLER] Updated load-balancing: " + loadb_.to_string() << std::endl;
  }

  void deploy(const std::string &app_id, Model &variant, Worker &worker)
  {
    variant.id = get_generator()->next();
    std::map<std::string, std::string> data = {
        {"id", std::to_string(variant.id)},
        {"name", variant.name},
        {"batch_size", std::to_string(variant.batch_size)},
    };
    Message msg("DEPLOY", data);
    send(worker, msg);
    if (datastore_.push(worker.get_id(), &variant) == nullptr)
    {
      std::cerr << "⛔️Error adding variant to worker" << std::endl;
    }
    std::cout << "👉[CONTROLLER] Will deploy " << variant.to_string() << " to " << worker.to_string() << std::endl;
  }

  void stop(const std::string &app_id, Model &variant, Worker &worker)
  {
    std::map<std::string, std::string> data = {
        {"id", std::to_string(variant.id)},
        {"name", variant.name},
    };
    Message msg("STOP", data);
    send(worker, msg);
    datastore_.remove(worker.get_id(), &variant);
    std::cout << "👉[CONTROLLER] Will stop " << variant.to_string() << " at " << worker.to_string() << std::endl;
  }

  void send(Worker &worker, const Message &msg)
  {
    networking_[worker.get_id()]->push(msg);
  }

  void query_daemon(const std::string &app_id)
  {
    std::cout << "😎Query forwarder will start for application " + app_id << std::endl;
    while (true)
    {
      std::optional<std::string> key = loadb_.next(app_id);

      if (key.has_value())
      {
        auto [variant, worker] = variant_worker_map_[key.value()];
        for (size_t i = 0; i < variant->batch_size; i++)
        {
          query_queue_[app_id].pop(); // blocking wait on a per-app queue
        }
        Message msg("QUERY", {{"variant_id", std::to_string(variant->id)}, {"batch_size", std::to_string(variant->batch_size)}});
        msg.append_data("variant_id", std::to_string(variant->id));
        send(*worker, msg);
        // std::cout << "⚪️Forward query" << std::endl;
      }
      else
      {
        std::cerr << "⚠️No variant instance found for the application " << app_id << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }

private:
  Event event_;
  LoadBalancer loadb_;
  Scheduler *scheduler_;
  AutoScaler *autoscaler_;
  Approach approach_;
  DataStore datastore_;
  InPort *incoming2_;
  std::map<int, OutPort *> networking_;
  // Queues and data
  std::unordered_map<std::string, BlockingQueue<Message>> query_queue_;
  BlockingQueue<Message> profiling_queue_;
  BlockingQueue<Message> registration_queue_;

  std::vector<std::thread> forward_query_threads_;
  std::unordered_map<std::string, std::pair<Model *, Worker *>> variant_worker_map_;
};

#endif // CONTROLLER_H