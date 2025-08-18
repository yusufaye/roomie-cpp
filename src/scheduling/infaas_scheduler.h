#ifndef INFAAS_SCHEDULER_H
#define INFAAS_SCHEDULER_H

#include <math.h>
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

class INFaaSScheduler : public Scheduler
{
public:
  INFaaSScheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates) override
  {
    return get_variant(workers, variant_candidates);
  }

  /**
   * This function takes the query's requirements, and the output is the variant and worker to serve the query.
    INFaaS considers variants in the Inactive state: getVariant first enquires the Metadata Store and retrieves the variant with the lowest combined loading and inference latency that matches the query's requirements (Line 5).
   */
  std::pair<Model *, Worker *> get_variant(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates)
  {
    // Autoscaler decides when to bring a worker up/down:
    // 1. When the utilization of any hardware resource exceeds a configurable threshold across all workers, the VMAutoscaler adds a new worker with the corresponding hardware resource. We empirically set the threshold to 80%, considering the time to instantiate VMs (20-30 seconds): a lower threshold triggers scaling too quickly and unnecessarily adds workers; a higher value may not scale in time.
    // 2. When variants on a particular hardware platform (e.g., GPU) are in the Interfered state across all workers, the VM-Autoscaler adds a worker with that hardware resource.
    // 3. When more than 80% of workers have Overloaded variants, the VM-Autoscaler starts a new worker.
    std::vector<Worker *> _workers;
    for (const auto worker : workers)
    {
      if (worker->get_total_running_variants() > 0 || _workers.empty())
      {
        _workers.push_back(worker);
      }
    }
    float percentage_of_overloaded_workers = 0.0;
    int number_overloaded_workers = 0;
    for (const auto worker : _workers)
    {
      for (Model *variant : worker->get_variants())
      {
        double throughput = variant->compute_throughput();
        double workload = variant->compute_workload();
        if ((workload / throughput) > 1.05)
        {
          number_overloaded_workers++;
          break;
        }
      }
    }
    if ((number_overloaded_workers / _workers.size()) > 0.8 && _workers.size() < workers.size())
    {
      _workers.push_back(workers[_workers.size()]); // Worker upscaling.
    }
    /**
     * INFaaS considers variants in the Inactive state: getVariant first enquires the Metadata Store and retrieves the variant with the lowest combined loading and inference latency that matches the query's requirements.
     * The available resources on the worker limit the number of variant instances it can run (Constraint #3).
     */
    std::vector<std::pair<Model *, Worker *>> current_workers;

    for (const auto &variant_name : variant_candidates)
    {
      for (auto *worker : _workers)
      {
        for (auto variant : worker->get_variants())
        {
          if (variant->name == variant_name)
          {
            for (int batch_size : BATCH_SIZES)
            {
              Model *new_variant = new Model(*variant);
              new_variant->batch_size = batch_size;
              if (new_variant->get_throughput() == 0 || worker->percent_occupation(new_variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
              {
                continue;
              }

              current_workers.emplace_back(new_variant, worker);
            }
          }
        }
      }
    }

    if (current_workers.empty())
    {
      for (const auto &variant_name : variant_candidates)
      {
        for (auto *worker : _workers)
        {
          Model *variant = this->load_model_metadata(worker->get_hardware_platform(), variant_name);
          for (int batch_size : BATCH_SIZES)
          {
            Model *new_variant = new Model(*variant);
            new_variant->batch_size = batch_size;
            if (new_variant->get_throughput() == 0 || worker->percent_occupation(new_variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
            {
              continue;
            }

            current_workers.emplace_back(new_variant, worker);
          }
        }
      }
    }

    /**
     * Thus, if the strategy requires more resources than are available on the current worker (e.g., hardware accelerator), the worker coordinates with the controller to load the variant on a capable worker.
     */
    // current_workers.clear();
    if (current_workers.empty())
    {
      for (const auto &variant_name : variant_candidates)
      {
        for (auto *worker : workers)
        {
          Model *variant = this->load_model_metadata(worker->get_hardware_platform(), variant_name);
          for (int batch_size : BATCH_SIZES)
          {
            Model *new_variant = new Model(*variant);
            new_variant->batch_size = batch_size;
            if (new_variant->get_throughput() == 0 || worker->percent_occupation(new_variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
            {
              continue;
            }

            current_workers.emplace_back(new_variant, worker);
          }
        }
      }
    }

    if (current_workers.empty())
    {
      spdlog::debug("🤕[INFaaS] Warning no variant candidate found for {}", variant_candidates[0]);
      return {nullptr, nullptr};
    }

    std::sort(current_workers.begin(), current_workers.end(),
              [](const std::pair<Model *, Worker *> a, const std::pair<Model *, Worker *> b)
              {
                if (a.first->get_throughput() != b.first->get_throughput())
                  return a.first->get_throughput() > b.first->get_throughput();
                return a.second->get_free_memory() > b.second->get_free_memory();
              });

    // Debug purpose
    int counter = 0;
    for (const auto item : workers)
    {
      if (item->get_total_running_variants() > 0)
        counter++;
    }
    return current_workers.front();
  }
};

#endif // INFAAS_SCHEDULER_H