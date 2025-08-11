#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <optional>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <condition_variable>
#include "engine.h"
#include "utils/general.h"
#include "utils/datastore.h"
#include "utils/load_balancing.h"
#include "networking/port.h"
#include "networking/message.h"
#include "scaling/auto_scaler.h"
#include "scheduling/base_scheduler.h"
#include "scheduling/usher_scheduler.h"
#include "scheduling/infaas_scheduler.h"
#include "scheduling/roomie_scheduler.h"

using namespace std::chrono;

enum class Approach
{
  ROOMIE,
  INFAAS,
  USHER
};

class Controller : public Engine
{
public:
  void configure(const json config)
  {
    Engine::configure(config);

    if (config_["parameters"]["scheduling"] == "INFaaSSchaduling")
    {
      scheduler_ = new INFaaSScheduler();
    }
    else if (config_["parameters"]["scheduling"] == "UsherSchaduling")
    {
      scheduler_ = new UsherScheduler();
    }
    else
    {
      scheduler_ = new RoomieScheduler();
    }

    incoming2_ = new InPort(get_incoming()[0]->get_host(), get_incoming()[0]->get_port() + 1, [this](Message msg)
                            { this->push(msg); });

    for (const auto &outport : outgoing_)
    {
      networking_[outport->getId()] = outport;
      Worker *worker = new Worker(outport->getId());
      worker->set_state(State::UNSET);
      datastore_.register_worker(worker);
    }
    const std::string logpath = config_["parameters"]["log_dir"].get<std::string>() + "/controller.csv";
    // Create a logger
    async_file_ = spdlog::basic_logger_mt<spdlog::async_factory>("async_file_logger", logpath, true);
    spdlog::warn("👉[controller] Controller will be saving log to {}", logpath);
    // Set a custom format string
    async_file_->set_pattern("%v");
    async_file_->set_level(spdlog::level::debug);
    async_file_->debug("{},{},{}", "timestamp", "in_sys_timestamp", "app_id");
  }

  void run() override
  {
    spdlog::debug("RUNNING CONTROLLER...");

    // Send HELLO messages to all outports
    for (const auto &worker : datastore_.get_workers())
    {
      Message msg("HELLO", {{"worker_id", worker->get_id()}});
      send(*worker, msg);
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

  void shutdown() override
  {
    Engine::shutdown();
    incoming2_->close();
  }

  void push(const Message &msg) override
  {
    // spdlog::info("✉️ [controller] Recv {}",  msg.to_string() );
    if (msg.get_type() == "REGISTER")
    {
      registration_queue_.push(msg);
    }
    else if (msg.get_type() == "QUERY")
    {
      query_queue_[msg.get_data()["app_id"]].push(msg);
      async_file_->debug("{},{},{}", duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count(), msg.get_timestamp(), msg.get_data()["app_id"].get<std::string>());
    }
    else if (msg.get_type() == "PROFILE_DATA")
    {
      profiling_queue_.push(msg);
    }
    else if (msg.get_type() == "HELLO")
    {
      try
      {
        int worker_id = msg.get_data()["worker_id"];
        double total_mem = msg.get_data()["total_mem"];
        std::string hardware_platform = msg.get_data()["hardware_platform"];
        Worker *worker = datastore_.get_worker(worker_id);
        worker->set_total_memory(total_mem / 2);
        worker->set_hardware_platform(hardware_platform);
        worker->set_state(State::SET);
        spdlog::debug("👉[controller] Update for {}", worker->to_string());
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto item : datastore_.get_workers())
        {
          if (item->get_state() == State::UNSET)
          {
            return;
          }
        }
        event_.set();
      }
      catch (const std::exception &e)
      {
        spdlog::error("⛔️ Error on hello replay\n\t{}", e.what());
      }
    }
    else if (msg.get_type() == "DEPLOYED")
    {
      try
      {
        int worker_id = msg.get_data()["worker_id"];
        double free_memory = msg.get_data()["free_memory"];
        Worker *worker = datastore_.get_worker(worker_id);
        worker->set_state(State::SET);
        spdlog::debug("👉[controller] Deployment done for {}", worker->to_string());
      }
      catch (const std::exception &e)
      {
        spdlog::error("⛔️ Error on deployed replay\n\t{}", e.what());
      }
    }
  }

  void registration_daemon()
  {
    try
    {
      event_.wait();
      for (const auto worker : datastore_.get_workers())
      {
        if (worker->get_state() != State::SET)
        {
          throw std::runtime_error("⛔️[controller] All workers are not set yet\n\t" + worker->to_string());
        }
      }
      spdlog::debug("👉[controller] About to run registration daemon");
      while (true)
      {
        Message msg = registration_queue_.pop(); // blocks until message arrives
        spdlog::debug("👉[controller] New registration {}", msg.to_string());
        std::map<std::string, std::string> variant_names = msg.get_data(); // assumed typed extraction
        int i(1);
        for (const auto [app_id, variant_name] : variant_names)
        {
          datastore_.register_app(variant_name, variant_name);
          std::vector<Worker *> workers = datastore_.get_workers();
          std::vector<std::string> names;
          for (const auto name : datastore_.get_registered(app_id))
          {
            names.push_back(name);
          }
          auto [variant, worker] = scheduler_->schedule(workers, names);
          if (variant == nullptr || worker == nullptr)
          {
            spdlog::warn("🤕[controller] Warning no variant candidate found for\n\t{}", app_id);
            continue;
          }
          deploy(app_id, *variant, *worker);

          forward_query_threads_.emplace_back([this, variant_name]()
                                              {
                                                query_daemon(variant_name); // starts query loop per variant
                                              });
          spdlog::debug("👉[controller] Registered app {} {}/{}", app_id, i++, variant_names.size());
        }

        autoscaler_->set_event();
      }
    }
    catch (const std::exception &e)
    {
      spdlog::error("⛔️ Error with registration daemon\n\t{}", e.what());
    }
  }

  void profiling_daemon()
  {
    try
    {
      while (true)
      {
        auto msg = profiling_queue_.pop(); // [TODO] Update load balancing weights.

        int worker_id = msg.get_data()["worker_id"];
        for (auto worker : datastore_.get_workers())
        {
          std::vector<json> items = msg.get_data()["variants"];
          for (const auto &item : items)
          {
            auto variant = datastore_.get_variant(item["variant_id"].get<int>());
            if (variant == nullptr)
            {
              spdlog::debug("No variant matches the given id {}", item["variant_id"].get<int>());
              continue;
            }
            variant->set_achieved_throughput(item["throughput"].get<float>());
            auto input_rates = item["input_rates"].get<std::vector<int>>();
            for (size_t i = 0; i < input_rates.size(); i++)
            {
              variant->input_rates[i] = input_rates[i];
            }
          }
        }
        update_load_balancer();
      }
    }
    catch (const std::exception &e)
    {
      spdlog::error("⛔️ Error with profiling daemon\n\t{}", e.what());
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
          spdlog::error("Warning: Zero throughput for variant {}", variant->id);
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
    // spdlog::debug("👉[controller] Updated load-balancing: {}", loadb_.to_string() );
  }

  void deploy(const std::string &app_id, Model &variant, Worker &worker)
  {
    if (worker.percent_occupation(variant.get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
    {
      throw std::runtime_error("⛔️[controller] error " + variant.to_string() + " to " + worker.to_string() + "\n\t| New occupancy: " + std::to_string(worker.percent_occupation(variant.get_memory())) + " (%)");
    }
    try
    {
      worker.set_state(State::DEPLOYING);
      variant.id = get_generator()->next();
      if (datastore_.push(worker.get_id(), &variant) == nullptr)
      {
        spdlog::error("⛔️ Error adding variant to worker");
      }
      Message msg("DEPLOY", {{"id", variant.id}, {"name", variant.name}, {"batch_size", variant.batch_size}});
      send(worker, msg);
      spdlog::debug("👉[controller] New deployment done for {}", app_id);
      int counter = 0;
      for (const auto worker: datastore_.get_workers())
      {
        if (worker->get_total_running_variants() > 0)
        {
          counter++;
        }
      }
      spdlog::warn("👉[controller] Used worker {}/{}", counter, datastore_.get_workers().size());
    }
    catch (const std::exception &e)
    {
      spdlog::error("⛔️ Error during deployment\n\t{}", e.what());
    }
  }

  void stop(const std::string &app_id, Model &variant, Worker &worker)
  {
    Message msg("STOP", {{"variant_id", variant.id}, {"variant_name", variant.name}});
    send(worker, msg);
    datastore_.remove(worker.get_id(), &variant);
    spdlog::debug("👉[controller] Will stop {} at {}", variant.to_string(), worker.to_string());
  }

  void send(Worker &worker, const Message &msg)
  {
    networking_[worker.get_id()]->push(msg);
  }

  void query_daemon(const std::string &app_id)
  {
    spdlog::debug("😎 Query forwarder will start for application " + app_id);
    while (true)
    {
      std::optional<std::string> key = loadb_.next(app_id);

      if (key.has_value())
      {
        auto [variant, worker] = variant_worker_map_[key.value()];
        double timestamp = 0.0;
        for (size_t i = 0; i < variant->batch_size; i++)
        {
          timestamp = query_queue_[app_id].pop().get_timestamp(); // blocking wait on a per-app queue
        }
        Message msg(timestamp, "QUERY", {{"variant_id", variant->id}, {"batch_size", variant->batch_size}});
        send(*worker, msg);
      }
      else
      {
        spdlog::debug("No variant instance found for the application {}", app_id);
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }

private:
  Event event_;
  std::mutex mutex_;
  std::shared_ptr<spdlog::logger> async_file_;
  LoadBalancer loadb_;
  Scheduler *scheduler_;
  AutoScaler *autoscaler_;
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