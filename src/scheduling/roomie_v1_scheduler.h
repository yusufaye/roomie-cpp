#ifndef ROOMIE_SCHEDULER_H
#define ROOMIE_SCHEDULER_H

#include <math.h>
#include <random>
#include <algorithm>
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

bool interfere(float prob)
{
  // Create a random number generator
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> uniformDis(0.0, 1.0);
  double randomNumber = uniformDis(gen);
  if (randomNumber < prob)
  {
    return true;
  }
  else
  {
    return false;
  }
}

std::vector<std::vector<double>> create_mask(const std::vector<double> &arr)
{
  int L = arr.size();
  int M = std::min(static_cast<int>(std::ceil(L / 2.0)), 5);
  M = (M % 2 == 0) ? M + 1 : M;
  std::vector<std::vector<double>> mask(M, std::vector<double>(L, 1.0));
  int window = M / 2;
  for (int pad = 1; pad <= window; ++pad)
  {
    for (int j = 0; j < pad; ++j)
    {
      mask[pad - 1][j] = 0.0; /* (f)-> i=0,1,...,w; j=0,1,...,w-1*/
    }
    for (int j = L - pad; j < L; ++j)
    {
      mask[M - pad][j] = 0.0; /* (b)-> i=N-w,N-w+1,...,N-1; j=N-w,N-w+1,...,N-1*/
    }
  }
  std::vector<std::vector<double>> result(M, std::vector<double>(L, 0.0));
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < L; ++j)
    {
      result[i][j] = arr[j] * mask[i][j];
    }
  }

  // e.g., of mask an array {1, 2, 3, 4, 5, 6, 7, 8}
  // [0 2 3 4 5 6 7 8 ]
  // |0 0 3 4 5 6 7 8 |
  // |1 2 3 4 5 6 7 8 |
  // |1 2 3 4 5 6 0 0 |
  // [1 2 3 4 5 6 7 0 ]
  return result;
}

class RoomieScheduler : public Scheduler
{
private:
  std::map<std::string, std::vector<float>> history_;

public:
  RoomieScheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates) override
  {
    std::vector<std::tuple<Model *, Worker *, std::vector<float>>> simulations;

    for (auto &variant_name : variant_candidates)
    {
      auto result = this->simulate(workers, variant_name); // simulate must be synchronous or handled via future/promise
      for (const auto &item : result)
      {
        simulations.push_back(item);
      }
    }

    if (!simulations.empty())
    {
      auto tmp = simulations;
      simulations.clear();
      for (const auto item : tmp)
      {
        // 1. Ensure that any variant doesn't have a perf drop beyond a certain threshold.
        auto it = max_element(std::get<2>(item).begin(), std::get<2>(item).end());
        if (*it <= 0.75)
        {
          simulations.push_back(item);
        }
        // 2. Or, ensure that on average the perf drop is under a certain threshold.
        // if (mean(std::get<2>(item)) <= 0.5)
        // {
        //   simulations.push_back(item);
        // }
      }
    }
    if (simulations.empty())
    {
      spdlog::debug("🤕[RoomieV1] Warning no variant candidate found");
      return {nullptr, nullptr};
    }

    std::sort(simulations.begin(), simulations.end(),
              [&](const std::tuple<Model *, Worker *, std::vector<float>> &a, const auto &b)
              {
                Model *variant = std::get<0>(a);
                float thr_a = (variant->get_throughput() - variant->get_throughput() * std::get<2>(a)[0]);
                variant = std::get<0>(b);
                float thr_b = (variant->get_throughput() - variant->get_throughput() * std::get<2>(b)[0]);
                return thr_a > thr_b;
              });

    auto &best = simulations.front();

    return {std::get<0>(best), std::get<1>(best)};
  }

  std::vector<std::tuple<Model *, Worker *, std::vector<float>>> simulate(
      const std::vector<Worker *> &workers,
      std::string &variant_name)
  {
    std::vector<std::tuple<Model *, Worker *, std::vector<float>>> results;

    for (Worker *worker : workers)
    {
      auto computations = this->compute(variant_name, worker);
      for (std::pair<Model *, std::vector<float>> item : computations)
      {
        results.push_back({item.first, worker, item.second});
      }
    }

    return results;
  }

  std::string build_key(const string &hardware_platform, const std::vector<Model *> &models)
  {
    std::vector<std::string> parts;
    for (const auto &item : models)
    {
      parts.push_back(item->name + "_" + std::to_string(item->batch_size));
    }

    std::sort(parts.begin(), parts.end());

    std::string key = parts[0];
    for (size_t i = 1; i < parts.size(); ++i)
    {
      key += "+" + parts[i];
    }

    return hardware_platform + "_" + key; // assuming Worker has to_string()
  }

  std::pair<std::vector<double>, std::vector<double>> heuristic_roomie(std::vector<Model *> &models, float prob = 0.2)
  {
    std::vector<double> durations;
    std::vector<double> new_durations;
    // std::vector<int> lengths;
    std::vector<std::vector<std::vector<double>>> masks;

    int N = models.size();
    for (int i = 0; i < N; ++i)
    {
      durations.push_back(models[i]->initial_duration());
      new_durations.push_back(durations[i]);
      // lengths.push_back(models[i]->get_kernels().size());
      std::vector<double> op_durations;
      std::vector<float> occ;
      for (auto &op : models[i]->get_kernels())
      {
        op_durations.push_back(op->duration);
      }
      masks.push_back(create_mask(op_durations));
    }

    for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < N; ++j) /* model to interfere with */
      {
        if (i == j)
        {
          continue;
        }

        auto mask_durations = masks[j];

        std::vector<double> delays;
        for (const auto &sample_dur : mask_durations)
        {
          double delayed = 0.0;
          for (size_t k = 0; k < mask_durations.size(); k++)
          {
            if (interfere(prob))
            {
              if (k > masks[i].size())
              {
                break;
              }
              delayed += sample_dur[k];
            }
          }
          delays.push_back(delayed);
        }
        /* Determine the number of time the model(j) might terminate while model(j) still executing. */
        double overlap = std::max((double)masks[i].size() / masks[j].size() / 2, 1.0);
        new_durations[i] += overlap * median(delays);
      }
    }

    return {durations, new_durations};
  }

  std::vector<std::pair<Model *, std::vector<float>>> compute(std::string &variant_name, Worker *&worker)
  {
    std::vector<std::pair<Model *, std::vector<float>>> results;
    Model *variant;
    for (int batch_size : BATCH_SIZES)
    {
      variant = new Model(*this->load_model_metadata(worker->get_hardware_platform(), variant_name));
      variant->batch_size = batch_size;

      if (worker->percent_occupation(variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY || variant->get_throughput() == 0)
      {
        continue;
      }

      std::vector<Model *> models;
      std::vector<float> perf_drops;

      if (worker->get_total_running_variants() > 0)
      {
        models.push_back(variant);
        for (const auto item : worker->get_variants())
        {
          models.push_back(item);
        }

        std::string key = this->build_key(worker->get_hardware_platform(), models);

        if (history_.find(key) != history_.end())
        {
          perf_drops = history_[key];
          results.push_back({variant, perf_drops});
          continue;
        }

        auto [durations, new_durations] = heuristic_roomie(models);

        if (new_durations < durations)
        {
          std::string oss = "Bad algorithms for ";
          for (auto *model : models)
          {
            oss += "(" + model->name + ", " + std::to_string(model->batch_size) + ") ";
          }

          oss += "\n\tDurations=[";
          for (size_t i = 0; i < durations.size(); i++)
          {
            oss += std::to_string(durations[i]) + ", ";
          }
          oss += "]\n\tNew durations=";
          for (size_t i = 0; i < new_durations.size(); i++)
          {
            oss += std::to_string(new_durations[i]) + ", ";
          }
          oss += "]";
          throw std::runtime_error(oss);
        }

        for (size_t i = 0; i < new_durations.size(); i++)
        {
          perf_drops.push_back((new_durations[i] - durations[i]) / new_durations[i]);
        }
        history_[key] = perf_drops;
      }
      else
      {
        perf_drops = {0.0};
      }
      results.push_back({variant, perf_drops});
    }

    return results;
  }
};

#endif // ROOMIE_SCHEDULER_H