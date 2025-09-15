#ifndef ROOMIE_V1_SCHEDULER_H
#define ROOMIE_V1_SCHEDULER_H

#include <math.h>
#include <random>
#include <algorithm>
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

class RoomieV1Scheduler : public Scheduler
{
private:
  std::map<std::string, std::vector<float>> history_;

public:
  RoomieV1Scheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates, bool scaling = false) override
  {
    std::string summary = "=== " + variant_candidates[0] + " ===";
    std::vector<std::tuple<Model *, Worker *, std::vector<float>>> simulations;

    for (auto &variant_name : variant_candidates)
    {
      auto result = this->simulate(workers, variant_name); // simulate must be synchronous or handled via future/promise
      for (const auto &item : result)
      {
        simulations.push_back(item);
      }
    }

    if (scaling && !simulations.empty())
    {
      for (auto it = simulations.begin(); it != simulations.end();)
      {
        // if (maximum(std::get<2>(*it)) > 0.5) /* 1. Ensure that any variant doesn't have a perf drop beyond a certain threshold. */
        if (mean(std::get<2>(*it)) > 0.5) /* 2. Or, ensure that on average the perf drop is under a certain threshold. */
        {
          it = simulations.erase(it);
        }
        else
        {
          ++it;
        }
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
                float thr_a = (variant->get_achieved_throughput() - variant->get_achieved_throughput() * std::get<2>(a)[0]);
                variant = std::get<0>(b);
                float thr_b = (variant->get_achieved_throughput() - variant->get_achieved_throughput() * std::get<2>(b)[0]);
                return thr_a > thr_b;
              });
    int N = std::min((int)simulations.size(), 5);
    for (size_t i = 0; i < N; i++)
    {
      const auto sim = simulations[i];
      summary += "\n N=" + std::to_string(std::get<1>(sim)->get_total_running_variants()) + " perf=" + vec2str(std::get<2>(sim));
    }
    spdlog::debug("{}", summary);

    auto &best = simulations.front();

    if (std::get<2>(best).size() > 1)
    {
      float current_throughput = 0.0;
      float next_throughput = std::get<0>(best)->get_achieved_throughput() - std::get<0>(best)->get_achieved_throughput() * std::get<2>(best)[0];
      int i = 1;
      for (const auto &variant : std::get<1>(best)->get_variants())
      {
        if (variant->name == std::get<0>(best)->name)
        {
          current_throughput += variant->get_achieved_throughput();
          next_throughput += variant->get_achieved_throughput() - variant->get_achieved_throughput() * std::get<2>(best)[i];
        }
        i++;
      }
      spdlog::debug("=== {} ===\n\tAchieved Throughput: {}\n\tPerf drop: {}\n\tCurrent Throughput: {}\n\tNext Throughput: {}", std::get<0>(best)->name, std::get<0>(best)->get_achieved_throughput(), vec2str(std::get<2>(best)), current_throughput, next_throughput);
      if (isnan(std::get<2>(best)[0]))
      {
        spdlog::debug("=== {}->{}: {} ===", std::get<0>(best)->name, std::get<0>(best)->batch_size, vec2str(std::get<2>(best)));
        for (const auto &variant : std::get<1>(best)->get_variants())
        {
          spdlog::debug("\t{}->{}", variant->name, variant->batch_size);
        }
      }
    }

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

  std::pair<std::vector<double>, std::vector<double>> heuristic_roomie(std::vector<Model *> &models)
  {
    std::vector<double> durations, new_durations;
    std::vector<int> lengths;
    std::vector<std::vector<std::vector<bool>>> masks;
    std::vector<std::vector<float>> occupancies;

    int N = models.size();

    if (N == 1)
    {
      auto duration = models[0]->initial_duration();
      durations.push_back(duration);
      return {durations, durations};
    }

    std::vector<std::vector<double>> model_kernel_durations;
    for (int i = 0; i < N; ++i)
    {
      auto duration = models[i]->initial_duration();
      durations.push_back(duration);
      new_durations.push_back(duration);
      auto size = models[i]->get_kernels().size();
      if (size == 0)
      {
        throw runtime_error("No kernel for the following " + models[i]->to_string());
      }
      lengths.push_back(size);
      std::vector<double> kernel_durations;
      std::vector<float> occ;
      for (auto &kernel : models[i]->get_kernels())
      {
        kernel_durations.push_back(kernel->duration);
        if (kernel->achieved_occupancy > 1)
        {
          occ.push_back(kernel->achieved_occupancy / 100);
        }
        else
        {
          occ.push_back(kernel->achieved_occupancy);
        }
      }
      occupancies.push_back(occ);
      masks.push_back(create_bool_mask(occ.size(), occ.size()));
      model_kernel_durations.push_back(kernel_durations);
    }

    for (int i = 0; i < N; ++i)
    {
      int counter = 0;
      for (int j = 0; j < N; ++j) /* model to interfere with */
      {
        if (i == j)
        {
          continue;
        }

        std::vector<double> additianal_durations;
        additianal_durations.resize(N, 0);
        std::vector<double> delays;
        for (const auto &model_mask : masks[j])
        {
          double delayed = 0.0;
          size_t k_i = 0;
          for (size_t k_j = 0; k_j < model_mask.size(); k_j++)
          {
            /* Early stopping is import when model with diffent size interfere. */
            if (k_i > lengths[i])
            {
              break;
            }
            if (model_mask[k_j])
            {
              if ((occupancies[i][k_i] + occupancies[j][k_j]) > 1.0)
              {
                std::vector<double> val = {model_kernel_durations[i][k_i], model_kernel_durations[j][k_j]};
                auto addition_duration = occupancies[i][k_i] / (occupancies[i][k_i] + occupancies[j][k_j]) * mean(val);

                delayed += addition_duration;
              }
              k_i++;
            }
          }
          delays.push_back(delayed);
        }
        /* Determine the number of time the model(j) might terminate while model(j) still executing. */
        auto overlap = std::max((float)lengths[i] / lengths[j], 1.0f);
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
      if (variant->get_kernels().size() == 0)
      {
        continue;
      }

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
          oss += "\n\tDurations: " + vec2str(durations) + "\n\tNew durations: " + vec2str(new_durations);
          throw std::runtime_error(oss);
        }

        for (size_t i = 0; i < new_durations.size(); i++)
        {
          perf_drops.push_back((new_durations[i] - durations[i]) / durations[i]);
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

#endif // ROOMIE_V1_SCHEDULER_H