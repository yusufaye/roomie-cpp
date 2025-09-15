#ifndef USHER_SCHEDULER_H
#define USHER_SCHEDULER_H

#include <math.h>
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

float Mreq(Model &variant, float total_memory)
{
  // Mreq of a model is the HIGHEST PERCENTAGE of the total memory space of a GPU consumed by the model at any point during its execution.

  return variant.get_memory() / total_memory * 100; // The highest percentage of memory.
}

float Creq(Model &variant)
{
  // Creq of a model is the HIGHEST PERCENTAGE of the total computation space of a GPU consumed by the model at any point during its execution.

  float total = 0.0;
  for (auto &it : variant.get_kernels())
    total += it->achieved_occupancy;
  return total / variant.get_kernels().size(); // The highest percentage of computation.
}

class UsherModel
{
public:
  Model *model;
  float c_req;
  float m_req;
  UsherModel(Model *model_, Worker *worker) : model(model_)
  {
    c_req = Creq(*model);
    m_req = Mreq(*model, worker->get_total_memory());
  }
};

bool Cheavy(UsherModel &variant, float threshold = 1.2)
{
  // A model is C-heavy if its average C-req/M-req ≥ 1.2.
  float ratio = variant.c_req / variant.m_req;
  return ratio >= threshold;
}

bool Mheavy(UsherModel &variant, float threshold = 1.2)
{
  // A model is M-heavy if M-req/C-req ≥ 1.2.
  float ratio = variant.m_req / variant.c_req;
  return ratio >= threshold;
}

class UsherScheduler : public Scheduler
{
public:
  UsherScheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates, bool scaling = false) override
  {
    return usher(workers, variant_candidates);
  }

  std::pair<Model *, Worker *> usher(std::vector<Worker *> workers, std::vector<string> variant_candidates)
  {
    std::vector<std::tuple<Model *, Worker *, float>> variant_worker_distances;
    for (const auto &variant_name : variant_candidates)
    {
      for (const int &batch_size : BATCH_SIZES)
      {
        for (const auto &worker : workers)
        {
          Model *new_variant = new Model(*this->load_model_metadata(worker->get_hardware_platform(), variant_name));
          new_variant->batch_size = batch_size;
          if (new_variant->get_throughput() > 0 && worker->percent_occupation(new_variant->get_memory()) <= MAX_GPU_MEMORY_OCCUPANCY)
          {
            int d = abs(Creq(*new_variant) - Mreq(*new_variant, worker->get_total_memory()));
            for (const auto &variant : worker->get_variants())
            {
              d += abs(Creq(*variant) - Mreq(*variant, worker->get_total_memory()));
            }
            variant_worker_distances.push_back({new_variant, worker, d});
          }
        }
      }
    }

    std::sort(variant_worker_distances.begin(), variant_worker_distances.end(), [](const std::tuple<Model *, Worker *, float> a, const std::tuple<Model *, Worker *, float> b)
              { return std::get<2>(a) < std::get<2>(b); });

    if (variant_worker_distances.empty())
    {
      spdlog::debug("🤕[Usher] Warning no variant candidate found");
      return {nullptr, nullptr};
    }

    auto best = variant_worker_distances.front();
    return {std::get<0>(best), std::get<1>(best)};
  }
};

#endif // USHER_SCHEDULER_H