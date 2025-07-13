#ifndef DATASTORE_H
#define DATASTORE_H

#include <map>
#include <mutex>
#include "NumCpp.hpp"
#include "kernels.h"

using namespace std;

class Model
{
private:
  float achieved_throughput = 0.0;
  map<int, float> map_throughput;
  map<int, unsigned long> map_memory;
  map<int, std::vector<NcuKernel *>> map_kernel;

public:
  int id;
  string name;
  string hardware_platform = "";
  int qsize = 0;
  float accuracy = 0.0;
  float occupancy = 0.0;
  int batch_size = 0;
  int model_memory = 0;
  int window_size = 10;
  nc::NdArray<int> input_rates = nc::zeros<int>(1, 10);

  Model() {}

  Model(const Model &model)
  {
    id = model.id;
    name = model.name;
    hardware_platform = model.hardware_platform;
    qsize = model.qsize;
    accuracy = model.accuracy;
    occupancy = model.occupancy;
    achieved_throughput = model.achieved_throughput;
    batch_size = model.batch_size;
    model_memory = model.model_memory;
    window_size = model.window_size;
    map_throughput = model.map_throughput;
    map_memory = model.map_memory;
    map_kernel = model.map_kernel;
  }

  Model(int id_, string name_, string hardware_platform_) : id(id_),
                                                            name(name_),
                                                            hardware_platform(hardware_platform_)
  {
  }

  std::vector<NcuKernel *> get_kernels(int bs = 0)
  {
    if (bs == 0)
      bs = batch_size;
    auto it = map_kernel.find(bs);
    if (it != map_kernel.end())
      return it->second;
    return std::vector<NcuKernel *>();
  }

  void set_kernels(vector<NcuKernel *> kernels, int batch_size)
  {
    map_kernel[batch_size] = kernels;
  }

  map<int, std::vector<NcuKernel *>> *get_Kernel()
  {
    return &map_kernel;
  }

  map<int, unsigned long> *get_Memory()
  {
    return &map_memory;
  }

  map<int, float> *get_Throughput()
  {
    return &map_throughput;
  }

  float get_profile_throughput()
  {
    return map_throughput[batch_size];
  }

  float input_rate()
  {
    return nc::mean(input_rates).item();
  }

  float initial_duration()
  {
    float value = 0;
    for (auto &it : get_kernels())
      value += it->duration;
    return value;
  }

  float get_throughput()
  {
    if (achieved_throughput > 0)
      return achieved_throughput;
    return get_profile_throughput();
  }

  void set_throughput(float achieved_throughput_)
  {
    achieved_throughput = achieved_throughput_;
  }

  unsigned long get_memory(int bs = 0)
  {
    if (bs == 0)
      bs = batch_size;
    return map_memory[bs];
  }

  float compute_workload()
  {
    // Calculate the workload based on the current load and the previously recorded input rate.
    return qsize + input_rates.sum().item();
  }

  float compute_throughput()
  {
    // Calculate the throughput based on the window size of the previously recorded input rate.
    return get_throughput() * input_rates.size();
  }

  void update(Model obj)
  {
    name = obj.name;
    qsize = obj.qsize;
    accuracy = obj.accuracy;
    occupancy = obj.occupancy;
    achieved_throughput = obj.achieved_throughput;
    batch_size = obj.batch_size;
    input_rates = obj.input_rates;
    hardware_platform = obj.hardware_platform;
  }

  bool operator==(const Model &c)
  {
    return id == c.id;
  }

  std::string to_string()
  {
    return "Model('id'=" + std::to_string(id) + ", 'name'=" + name + ", 'thr'=" + std::to_string(get_throughput()) + ", 'bs'=" + std::to_string(batch_size) + ", 'mem'=" + std::to_string(get_memory()) + ")";
  }

  ostream &operator<<(ostream &os)
  {
    return os << "Model(id=" << id << ", name=" << name << ", thr=" << get_throughput() << ", bs=" << batch_size << ", mem=" << get_memory() << ")";
  }
};

class Worker
{
public:
  Worker(int id, int device = 0, string hardware_platform = "xavier")
      : id(id), device(device), hardware_platform(hardware_platform), total_memory(0.0) {}

  float get_free_memory() const
  {
    float total_variant_memory = 0.0f;
    for (const auto &variant : variants)
    {
      total_variant_memory += variant->get_memory();
    }
    return total_memory - total_variant_memory;
  }

  float percent_occupation(float additional = 0.0f) const
  {
    float mem_used = additional;
    for (const auto &variant : variants)
    {
      mem_used += variant->get_memory();
    }
    return (mem_used / total_memory) * 100.0f;
  }

  void update_variant(Model &variant)
  {
    for (auto &running_variant : variants)
    {
      if (*running_variant == variant)
      {
        running_variant->update(variant);
        return;
      }
    }
  }

  int get_total_running_variants() const
  {
    return variants.size();
  }

  bool operator==(const Worker &other) const
  {
    return id == other.id;
  }

  std::string to_string()
  {
    std::string os = "Worker('id'=" + std::to_string(id) + ", 'free memory'=" + std::to_string(get_free_memory()) + ", 'total memory'=" + std::to_string(total_memory) + ", 'hardware platform'=" + hardware_platform + ", 'variants'=[";
    for (size_t i = 0; i < variants.size(); ++i)
    {
      os += variants[i]->name;
      if (i < variants.size() - 1)
      {
        os += ", ";
      }
    }
    os += "])";
    return os;
  }

  friend ostream &operator<<(ostream &os, const Worker &worker)
  {
    os << "Worker('id'=" << worker.id << ", 'free memory'=" << worker.get_free_memory()
       << ", 'total memory'=" << worker.total_memory << ", 'hardware platform'=" << worker.hardware_platform
       << ", 'variants'=[";
    for (size_t i = 0; i < worker.variants.size(); ++i)
    {
      os << worker.variants[i]->name;
      if (i < worker.variants.size() - 1)
      {
        os << ", ";
      }
    }
    os << "])";
    return os;
  }

  void set_total_memory(double value) { total_memory = value; }

  void add_variant(Model *variant)
  {
    variants.push_back(variant);
  }

  void remove_variant(Model *variant)
  {
    for (auto it = variants.begin(); it != variants.end(); ++it)
    {
      if (**it == *variant)
      {
        variants.erase(it);
        break;
      }
    }
  }

  std::vector<Model *> get_variants() const { return variants; }
  int get_id() const { return id; }
  int get_device() const { return device; }
  double get_total_memory() const { return total_memory; }
  string get_hardware_platform() const { return hardware_platform; }

private:
  int id;
  int device;
  string hardware_platform;
  double total_memory;
  string device_name;
  std::vector<Model *> variants;
};

class DataStore
{
private:
  std::map<std::string, std::set<string>> registration_;
  std::vector<Worker *> workers_;
  std::mutex mutex_;

public:
  DataStore() {}

  std::map<std::string, std::set<string>> get_registration() const { return registration_; }

  std::vector<Worker *> &get_workers()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return workers_;
  }

  void register_app(const string &app_id, const string &variant_name)
  {
    registration_[app_id].insert(variant_name);
  }

  std::map<std::string, std::set<string>> &get_registration()
  {
    return registration_;
  }

  set<string> get_registered(const string &app_id) const
  {
    auto it = registration_.find(app_id);
    if (it != registration_.end())
    {
      return it->second;
    }
    return {};
  }

  std::vector<Model *> get_variants(const string &app_id)
  {
    std::vector<Model *> variants;
    for (auto &worker : get_workers())
    {
      for (const auto &variant : worker->get_variants())
      {
        if (registration_.at(app_id).find(variant->name) != registration_.at(app_id).end())
        {
          variants.push_back(variant);
        }
      }
    }
    return variants;
  }

  std::vector<Model *> get_variants()
  {
    std::vector<Model *> variants;
    for (auto &worker : get_workers())
    {
      for (const auto &variant : worker->get_variants())
      {
        variants.push_back(variant);
      }
    }
    return variants;
  }

  void register_worker(Worker *worker)
  {
    workers_.push_back(worker);
  }

  Worker *push(int id, Model *variant)
  {
    for (auto &worker : get_workers())
    {
      if (worker->get_id() != id)
        continue;
      worker->add_variant(variant);
      return worker;
    }
    return nullptr;
  }

  void remove(int id, Model *variant)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &worker : workers_)
    {
      if (worker->get_id() != id)
        continue;
      worker->remove_variant(variant);
      break;
    }
  }

  void update(Worker *worker)
  {
    auto it = find(workers_.begin(), workers_.end(), worker);
    if (it != workers_.end())
    {
      int idx = distance(workers_.begin(), it);
      for (auto &variant : worker->get_variants())
      {
        if (variant == nullptr)
          continue;
        workers_[idx]->set_total_memory(worker->get_total_memory());
        workers_[idx]->update_variant(*variant);
      }
    }
    else
    {
      throw invalid_argument("Worker not found");
    }
  }

  std::vector<std::pair<Model *, Worker *>> get_variant_workers()
  {
    static std::vector<std::pair<Model *, Worker *>> variant_workers;
    for (auto &worker : get_workers())
    {
      for (const auto &variant : worker->get_variants())
      {
        variant_workers.emplace_back(variant, worker);
      }
    }
    return variant_workers;
  }

  std::vector<std::pair<Model *, Worker *>> get_variant_workers(const std::string &app_id)
  {
    std::vector<std::pair<Model *, Worker *>> variant_workers;
    for (auto &worker : get_workers())
    {
      for (const auto &variant : worker->get_variants())
      {
        if (registration_.at(app_id).find(variant->name) != registration_.at(app_id).end())
        {
          variant_workers.emplace_back(variant, worker);
        }
      }
    }
    return variant_workers;
  }

  Worker *get_worker(int id)
  {
    for (auto &worker : get_workers())
    {
      if (worker->get_id() == id)
      {
        return worker;
      }
    }
    throw std::runtime_error("❌No worker matches the given id " + std::to_string(id));
  }

  // friend ostream &operator<<(ostream &os, const DataStore &data_store)
  // {
  //   for (const auto &worker : data_store.workers)
  //   {
  //     os << *worker << "\n";
  //   }
  //   return os;
  // }
};

#endif // DATASTORE_H