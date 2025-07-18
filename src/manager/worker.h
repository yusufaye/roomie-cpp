#ifndef WORKER_ENGINE_H
#define WORKER_ENGINE_H

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <torch/script.h>
#include <cuda_runtime.h>
#include <condition_variable>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "engine.h"
#include "utils/queue.h"
#include "utils/datastore.h"
#include "networking/port.h"
#include "networking/message.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class WorkerEngine : public Engine
{
public:
  void configure(const json config)
  {
    Engine::configure(config);
    device_ = new torch::Device(torch::kCUDA, device_index_);
    device_index_ = config_["parameters"]["device"].get<int>();
    hardware_platform_ = config_["parameters"]["hardware_platform"];
    std::cout << "👉[WORKER] Given device is " << device_index_ << " 👈" << std::endl;
  }

  void run() override
  {
    std::cout << "RUNNING WORKER..." << std::endl;
    std::thread registration_thread(&WorkerEngine::deployment_daemon, this);
    std::thread monitor_thread(&WorkerEngine::monitor_daemon, this);
    std::thread monitor_incoming_thread(&WorkerEngine::monitor_incoming_data, this);
    registration_thread.join();
    monitor_thread.join();
    monitor_incoming_thread.join();
  }

  void push(const Message &msg) override
  {
    // std::cout << "👉[WORKER] Recv: " << msg.to_string() << std::endl;
    if (msg.getType() == "DEPLOY")
    {
      deployment_queue_.push(msg);
    }
    else if (msg.getType() == "QUERY")
    {
      if (inference_queue_.find(msg.get_data()["variant_id"]) != inference_queue_.end())
      {
        inference_queue_[msg.get_data()["variant_id"]]->push(1); // [TODO] push actual data.
        num_received_[msg.get_data()["variant_id"]] += std::stoi(msg.get_data()["batch_size"]);
      }
    }
    else if (msg.getType() == "STOP")
    {
      inference_queue_[msg.get_data()["variant_id"]]->push(0);
    }
    else if (msg.getType() == "HELLO")
    {
      std::cout << "👉[WORKER] Hello messge received: " << msg.to_string() << std::endl;
      id_ = std::stoi(msg.get_data()["worker_id"]);
      engine_name_ = "WorkerEngine-" + msg.get_data()["worker_id"];
      size_t free_memory;
      size_t total_memory;
      cudaMemGetInfo(&free_memory, &total_memory);
      std::cout << "Total memory: " << total_memory / (1024.0 * 1024) << " MB | Free memory: " << free_memory / (1024.0 * 1024) << " MB" << std::endl;
      total_mem_ = total_memory;
      Message hello_msg("HELLO", {{"worker_id", msg.get_data()["worker_id"]}, {"total_mem", std::to_string(total_mem_)}});
      outgoing_[0]->push(hello_msg);
    }
  }

  void monitor_incoming_data()
  {
    std::map<std::string, int> input_rate;
    while (true)
    {
      input_rate.clear();
      for (const auto [variant_id, num_received] : num_received_)
      {
        input_rate[variant_id] = num_received;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
      for (auto [variant_id, num_received] : num_received_)
      {
        int size(running_variant_[variant_id]->input_rates.size());
        // Shift elements to the right
        for (int i = size - 1; i > 0; --i)
        {
          running_variant_[variant_id]->input_rates[i] = running_variant_[variant_id]->input_rates[i - 1];
        }
        running_variant_[variant_id]->input_rates[0] = num_received - input_rate[variant_id];
      }
    }
  }

  void monitor_daemon()
  {
    try
    {
      while (true)
      {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::vector<std::string> items;
        json j;
        for (const auto [_, variant] : running_variant_)
        {
          j.push_back({
              {"variant_id", variant->id},
              {"variant_name", variant->name},
              {"throughput", variant->get_throughput()},
              {"input_rate", variant->input_rates},
          });
        }
        std::map<std::string, std::string> data = {{"worker_id", std::to_string(id_)}, {"variants", j.dump()}};
        Message msg("PROFILE_DATA", data);
        outgoing_[0]->push(msg);
        // std::cout << "👉[WORKER] Monitoring with " + msg.to_string() << std::endl;
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️ Error with monitor daemon\n\t" << e.what() << '\n';
    }
  }

  void deployment_daemon()
  {
    try
    {
      while (true)
      {
        auto msg = deployment_queue_.pop(); // blocks until message arrives
        // std::cout << "👉[WORKER] About to deploy " << msg.to_string() << std::endl;
        Model *model = new Model();
        model->id = std::stoi(msg.get_data()["id"]);
        model->name = msg.get_data()["name"];
        model->batch_size = std::stoi(msg.get_data()["batch_size"]);
        running_variant_[msg.get_data()["id"]] = model;
        num_received_[msg.get_data()["id"]] = 0;
        auto queue = new BlockingQueue<int>();
        inference_queue_[msg.get_data()["id"]] = queue;
        inference_threads_.emplace_back([this, model, queue]()
                                        { run_inference(model, queue); });
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️ Error with profiling daemon\n\t" << e.what() << '\n';
    }
  }

  void run_inference(Model *model, BlockingQueue<int> *queue)
  {
    try
    {
      string model_filename = "src/data/models/" + model->name + ".pt";

      c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(/* isHighPriority = */ true, /* device_index = */ 0);
      c10::cuda::setCurrentCUDAStream(stream);
      torch::jit::script::Module module = torch::jit::load(model_filename);
      module.to(torch::kCUDA);

      torch::Tensor input = torch::randn({model->batch_size, 3, 224, 224}, torch::device(torch::kCUDA));
      int data;
      chrono::_V2::system_clock::time_point startTime;
      chrono::_V2::system_clock::time_point endTime;

      size_t free_memory;
      size_t total_memory;
      cudaMemGetInfo(&free_memory, &total_memory);
      std::cout << "⚠️ [worker] New deployment\n\t| Name: " + model->name + "\n\t| Batch-size: " + std::to_string(model->batch_size) + "\n\t| Free-memory: " + std::to_string(free_memory / (1024.0 * 1024)) + " MB" << std::endl;
      Message msg("DEPLOYED", {{"worker_id", std::to_string(id_)}, {"free_memory", std::to_string(free_memory)}, {"total_memory", std::to_string(total_memory)}});
      outgoing_[0]->push(msg);
      while (true)
      {
        try
        {
          data = queue->pop();
          if (data == 0)
          {
            std::cout << "⚠️ [worker] About to stop | Name: " + model->name + ", batch-size: " + std::to_string(model->batch_size) << std::endl;
            return;
          }
          startTime = chrono::high_resolution_clock::now();
          module.forward({input});
          // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // [TODO] Debug purpose.
          endTime = chrono::high_resolution_clock::now();
          model->set_throughput(model->batch_size / std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime).count());
        }
        catch (const std::exception &e)
        {
          std::cerr << e.what() << '\n';
        }
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "⛔️ Error initializing model\n\t" << e.what() << '\n';
    }
  }

private:
  Event event_;
  // Queues and data
  std::map<std::string, BlockingQueue<int> *> inference_queue_;
  std::map<std::string, int> num_received_;
  BlockingQueue<Message> deployment_queue_;

  std::map<std::string, Model *> running_variant_;
  std::vector<std::thread> inference_threads_;
  std::string hardware_platform_;
  torch::Device *device_;
  int device_index_;
  double total_mem_;
};

#endif // WORKER_ENGINE_H