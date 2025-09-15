#ifndef LESS_RUNNING_VARIANTS_SCHEDULER_H
#define LESS_RUNNING_VARIANTS_SCHEDULER_H

#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

class LessRunningVariantsScheduler : public Scheduler
{
public:
  LessRunningVariantsScheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates, bool scaling=false) override
  {
    Worker *worker = workers.front();
    for (const auto _worker : workers)
    {

      if (_worker->get_total_running_variants() < worker->get_total_running_variants())
      {
        worker = _worker;
      }
      else
      {
        if (_worker->get_free_memory() > worker->get_free_memory())
        {
          worker = _worker;
        }
      }
    }
    std::vector<std::pair<Model *, Worker *>> variant_workers;
    for (const auto &variant_name : variant_candidates)
    {
      for (int batch_size : BATCH_SIZES)
      {
        Model *new_variant = new Model(*load_model_metadata(worker->get_hardware_platform(), variant_name));
        new_variant->batch_size = batch_size;
        if (new_variant->get_throughput() == 0 || worker->percent_occupation(new_variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
        {
          continue;
        }

        variant_workers.emplace_back(new_variant, worker);
      }
    }

    if (variant_workers.empty())
    {
      spdlog::debug("🤕[LessRunningVariants] Warning no variant candidate found for {}", variant_candidates[0]);
      return {nullptr, nullptr};
    }

    std::sort(variant_workers.begin(), variant_workers.end(),
              [](const std::pair<Model *, Worker *> a, const std::pair<Model *, Worker *> b)
              {
                return a.first->get_throughput() > b.first->get_throughput();
              });

    return variant_workers.front();
  }
};

#endif // LESS_RUNNING_VARIANTS_SCHEDULER_H