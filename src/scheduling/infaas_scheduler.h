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

  std::pair<Model *, Worker *> get_variant(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates)
  {
    std::vector<std::pair<Model *, Worker *>> current_workers;

    for (const auto &variant_name : variant_candidates)
    {
      for (auto *worker : workers)
      {
        Model *variant;
        try
        {
          variant = this->load_model_metadata(worker->get_hardware_platform(), variant_name);
        }
        catch (const std::exception &e)
        {
          std::cerr << "⛔️Error while load model metadata\n\t" << e.what() << '\n';
          throw e;
        }

        // std::vector<std::string> names;
        // for (auto *_variant : worker->get_variants())
        // {
        //   names.push_back(_variant->name);
        // }

        // if (std::find(names.begin(), names.end(), variant->name) == names.end())
        //   continue;

        for (int batch_size : BATCH_SIZES)
        {
          Model *new_variant = new Model(*variant);
          new_variant->batch_size = batch_size;
          if (new_variant->get_profile_throughput() == 0 ||
              worker->percent_occupation(new_variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
          {
            continue;
          }

          current_workers.emplace_back(new_variant, worker);
        }
      }
    }

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

            if (new_variant->get_profile_throughput() == 0 ||
                worker->percent_occupation(new_variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
            {
              std::cout << "--> Cannot hold " << new_variant->to_string() << " Free mem: " << worker->percent_occupation(new_variant->get_memory()) << "(" << MAX_GPU_MEMORY_OCCUPANCY << "%)" << std::endl;
              continue;
            }

            current_workers.emplace_back(new_variant, worker);
          }
        }
      }
    }

    if (current_workers.empty())
    {
      std::cout << "🤕[INFaaS] Warning no variant candidate found for " + variant_candidates[0] << std::endl;
      return {nullptr, nullptr};
    }

    std::sort(current_workers.begin(), current_workers.end(),
              [](const std::pair<Model *, Worker *> a, const std::pair<Model *, Worker *> b)
              {
                if (a.first->get_profile_throughput() != b.first->get_profile_throughput())
                  return a.first->get_profile_throughput() > b.first->get_profile_throughput();
                return a.second->get_free_memory() > b.second->get_free_memory();
              });

    return current_workers.front();
  }
};

#endif // INFAAS_SCHEDULER_H