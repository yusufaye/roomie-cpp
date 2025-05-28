#ifndef DATASTORE_H
#define DATASTORE_H

#include "NumCpp.hpp"
#include <map>
#include "kernels.h"

using namespace std;

class Model
{
private:
  map<int, float> map_throughput;
  map<int, int> map_memory;
  map<int, vector<NcuKernel *>> map_kernel;

public:
  int id;
  string name;
  int qsize = 0;
  bool warming = false;
  float accuracy = 0.0;
  float occupancy = 0.0;
  float throughput = 0.0;
  int batch_size = 0;
  int model_memory = 0;
  int window_size = 10;
  nc::NdArray<float> input_rates = nc::zeros<float>(1, 10);
  string hardware_platform = "";

  Model(int id_, string name_, string hardware_platform_) : id(id_),
                                                            name(name_),
                                                            hardware_platform(hardware_platform_)
  {
  }

  vector<NcuKernel *> get_kernels(int bs = 0)
  {
    if (bs == 0)
      bs = batch_size;
    auto it = map_kernel.find(bs);
    if (it != map_kernel.end())
      return it->second;
    return vector<NcuKernel *>();
  }

  void set_kernels(vector<NcuKernel *> kernels, int batch_size)
  {
    map_kernel[batch_size] = kernels;
  }

  map<int, vector<NcuKernel *>> *get_Kernel()
  {
    return &map_kernel;
  }

  map<int, int> *get_Memory()
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
  
  float achieved_throughput()
  {
    if (throughput > 0)
      return throughput;
    return get_profile_throughput();
  }

  float get_memory(int bs=0) {
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
    return achieved_throughput() * input_rates.size();
  }

  void update(Model obj)
  {
    name = obj.name;
    qsize = obj.qsize;
    warming = obj.warming;
    accuracy = obj.accuracy;
    occupancy = obj.occupancy;
    throughput = obj.throughput;
    batch_size = obj.batch_size;
    input_rates = obj.input_rates;
    hardware_platform = obj.hardware_platform;
  }

  bool operator==(const Model &c)
  {
    return id == c.id;
  }

  ostream &operator<<(ostream &os)
  {
    return os << "Model(id=" << id << ", name=" << name << ", thr=" << throughput << ", bs=" << batch_size << ", mem=" << get_memory() << ")";
  }
};

class Worker
{
public:
  Worker(int id, int device = 0, string hardware_platform = "xavier")
      : id(id), device(device), hardware_platform(hardware_platform), total_memory(0.0), warming(false), _nvidia_gpu_spec(nullptr) {}

  NvidiaGpuSpec *get_nvidia_gpu_spec() const
  {
    return _nvidia_gpu_spec;
  }

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

  friend ostream &operator<<(ostream &os, const Worker &worker)
  {
    os << "Worker('id'=" << worker.id << ", 'free memory'=" << worker.get_free_memory()
       << ", 'total memory'=" << worker.total_memory << ", 'hardware platform'=" << worker.hardware_platform
       << ", 'warming'=" << worker.is_warming() << ", 'variants'=[";
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

  void set_warming(bool value) { warming = value; }
  void set_total_memory(float value) { total_memory = value; }

  vector<Model *> get_variants() { return variants; }
  int get_id() const { return id; }
  int get_device() const { return device; }
  float get_total_memory() const { return total_memory; }
  bool is_warming() const { return warming == true; }
  string get_hardware_platform() const { return hardware_platform; }

private:
  int id;
  int device;
  string hardware_platform;
  float total_memory;
  bool warming;
  string device_name;
  vector<Model *> variants;
  NvidiaGpuSpec *_nvidia_gpu_spec;
};

class DataStore
{
private:
  map<string, set<string>> registration;
  vector<Worker *> workers;

public:
  DataStore() {}

  void register_app(const string &app_id, const string &variant_name)
  {
    registration[app_id].insert(variant_name);
  }

  map<string, set<string>> get_registration()
  {
    return registration;
  }

  set<string> get_registered(const string &app_id) const
  {
    auto it = registration.find(app_id);
    if (it != registration.end())
    {
      return it->second;
    }
    return {};
  }

  vector<Model *> get_variants(const string &app_id) const
  {
    vector<Model *> variants;
    for (const auto &worker : workers)
    {
      for (const auto &variant : worker->get_variants())
      {
        if (registration.at(app_id).find(variant->name) != registration.at(app_id).end())
        {
          variants.push_back(variant);
        }
      }
    }
    return variants;
  }

  void register_worker(Worker *worker)
  {
    workers.push_back(worker);
  }

  Worker *push(int id, Model *variant)
  {
    for (auto &worker : workers)
    {
      if (worker->get_id() != id)
        continue;
      for (auto &v : worker->get_variants())
      {
        if (*v == *variant)
        {
          worker->update_variant(*variant);
          break;
        }
      }
      worker->get_variants().push_back(variant);
      worker->set_warming(true);
      return worker;
    }
    return nullptr;
  }

  void remove(int id, Model *variant)
  {
    for (auto &worker : workers)
    {
      if (worker->get_id() != id)
        continue;
      for (auto it = worker->get_variants().begin(); it != worker->get_variants().end(); ++it)
      {
        if (**it == *variant)
        {
          worker->get_variants().erase(it);
          break;
        }
      }
      break;
    }
  }

  pair<map<string, set<string>>, vector<Worker *>> update(Worker *worker)
  {
    auto it = find(workers.begin(), workers.end(), worker);
    if (it != workers.end())
    {
      int idx = distance(workers.begin(), it);
      for (auto &variant : worker->get_variants())
      {
        if (variant == nullptr)
          continue;
        workers[idx]->set_total_memory(worker->get_total_memory());
        workers[idx]->update_variant(*variant);
      }
      return get_state();
    }
    else
    {
      throw invalid_argument("Worker not found");
    }
  }

  void warming_done(Worker *worker)
  {
    auto it = find(workers.begin(), workers.end(), worker);
    if (it != workers.end())
    {
      int idx = distance(workers.begin(), it);
      workers[idx]->set_warming(false);
    }
  }

  pair<map<string, set<string>>, vector<Worker *>> get_state() const
  {
    return {registration, workers};
  }

  vector<pair<Model *, Worker *>> get_variant_workers(const string &app_id) const
  {
    vector<pair<Model *, Worker *>> variant_workers;
    for (const auto &worker : workers)
    {
      for (const auto &variant : worker->get_variants())
      {
        if (registration.at(app_id).find(variant->name) != registration.at(app_id).end())
        {
          variant_workers.emplace_back(variant, worker);
        }
      }
    }
    return variant_workers;
  }

  vector<Worker *> get_workers()
  {
    return workers;
  }

  Worker *get_worker(int id)
  {
    for (const auto &worker : workers)
    {
      if (worker->get_id() == id)
      {
        return worker;
      }
    }
    return nullptr;
  }

  friend ostream &operator<<(ostream &os, const DataStore &data_store)
  {
    for (const auto &worker : data_store.workers)
    {
      os << *worker << "\n";
    }
    return os;
  }

  map<string, set<string>> get_registration() const { return registration; }
  vector<Worker *> get_workers() const { return workers; }
};

#endif // DATASTORE_H