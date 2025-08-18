#ifndef INFAAS_SCHEDULER_H
#define INFAAS_SCHEDULER_H

#include <math.h>
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

class INFaaSV1Scheduler : public Scheduler
{
public:
  INFaaSV1Scheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates) override
  {
    return get_variant(workers, variant_candidates);
  }

  /**
   * This function takes the query's requirements, and the output is the variant and worker to serve the query.
    INFaaSV1 considers variants in the Inactive state: getVariant first enquires the Metadata Store and retrieves the variant with the lowest combined loading and inference latency that matches the query's requirements (Line 5).
   */
  std::pair<Model *, Worker *> get_variant(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates)
  {
    /**
     * INFaaSV1 considers variants in the Inactive state: getVariant first enquires the Metadata Store and retrieves the variant with the lowest combined loading and inference latency that matches the query's requirements.
     * The available resources on the worker limit the number of variant instances it can run (Constraint #3).
     */
    std::vector<std::pair<Model *, Worker *>> current_workers;

    for (const auto &variant_name : variant_candidates)
    {
      for (auto *worker : workers)
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

    /**
     * Thus, if the strategy requires more resources than are available on the current worker (e.g., hardware accelerator), the worker coordinates with the controller to load the variant on a capable worker.
     */
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
      spdlog::debug("🤕[INFaaSV1] Warning no variant candidate found for {}", variant_candidates[0]);
      return {nullptr, nullptr};
    }

    std::sort(current_workers.begin(), current_workers.end(),
              [](const std::pair<Model *, Worker *> a, const std::pair<Model *, Worker *> b)
              {
                if (a.first->get_throughput() != b.first->get_throughput())
                  return a.first->get_throughput() > b.first->get_throughput();
                return a.second->get_free_memory() > b.second->get_free_memory();
              });

    return current_workers.front();
  }
};

#endif // INFAAS_SCHEDULER_H