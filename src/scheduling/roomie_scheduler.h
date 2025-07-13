#ifndef ROOMIE_SCHEDULER_H
#define ROOMIE_SCHEDULER_H

#include <math.h>
#include "NumCpp.hpp"
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

nc::NdArray<double> create_mask(const nc::NdArray<double> &arr)
{
  int L = arr.size();
  int M = std::min(static_cast<int>(std::ceil(L / 2.0)), 5);
  M = (M % 2 == 0) ? M + 1 : M;

  nc::NdArray<double> mask = nc::ones<double>(3, 4);

  int max_pad = M / 2;
  for (int pad = 1; pad <= max_pad; ++pad)
  {
    mask(pad - 1, nc::Slice(0, L - pad)) = 0;
    mask(M - pad, nc::Slice(L - pad, L)) = 0;
  }

  nc::NdArray<double> result = arr * mask;

  return result;
}

class RoomieScheduler : public Scheduler
{
public:
  RoomieScheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates) override
  {
    std::vector<std::tuple<Model *, Worker *, std::vector<float>>> simulations;

    for (auto &variant_name : variant_candidates)
    {
      auto result = this->simulate(workers, variant_name); // simulate must be synchronous or handled via future/promise
      simulations.insert(simulations.end(), result.begin(), result.end());
    }

    auto sort_fn = [](const std::tuple<Model *, Worker *, std::vector<float>> value)
    {
      const auto &metrics = std::get<2>(value);
      return *std::max_element(metrics.begin(), metrics.end());
    };

    std::sort(simulations.begin(), simulations.end(),
              [&](const auto &a, const auto &b)
              {
                return sort_fn(a) < sort_fn(b);
              });

    if (simulations.empty())
    {
      std::cout << "[Roomie] Warning no variant candidate found" << std::endl;
      return {nullptr, nullptr};
    }

    auto &best = simulations.front();
    return {std::get<0>(best), std::get<1>(best)};
  }

  std::vector<std::tuple<Model *, Worker *, std::vector<float>>> simulate(
      const std::vector<Worker *> &workers,
      std::string &variant_name)
  {
    std::vector<std::tuple<Model *, Worker *, std::vector<float>>> results;

    for (auto *worker : workers)
    {
      auto computations = this->compute(variant_name, worker);
      for (const auto &[variant, perf_drops] : computations)
      {
        results.emplace_back(variant, worker, perf_drops);
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

  std::pair<nc::NdArray<double>, nc::NdArray<double>> heuristic_roomie(std::vector<Model *> &models)
  {
    int N = models.size();
    nc::NdArray<double> durations(N);
    nc::NdArray<double> new_durations(N);
    std::vector<int> lengths;
    std::vector<nc::NdArray<double>> masks;

    for (int i = 0; i < N; ++i)
    {
      durations[i] = models[i]->initial_duration();
      new_durations[i] = durations[i];
      lengths.push_back(models[i]->get_kernels().size());

      std::vector<double> op_durations;
      for (const auto &op : models[i]->get_kernels())
        op_durations.push_back(op->duration);

      nc::NdArray<double> op_array = nc::NdArray<double>(op_durations);
      masks.push_back(create_mask(op_array));
    }

    for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < N; ++j)
      {
        if (i == j)
          continue;

        auto mask_durations = masks[j];
        int p = static_cast<int>(std::ceil((double)lengths[i] / lengths[j] / 2));

        auto probs = nc::random::rand<double>({mask_durations.shape().rows, mask_durations.shape().cols});
        auto bool_mask = probs < 0.8;

        std::vector<double> sums;

        for (int k = 0; k < mask_durations.shape().rows; ++k)
        {
          double sum = 0.0;
          for (int l = 0; l < mask_durations.shape().cols; ++l)
          {
            if (bool_mask(k, l))
              sum += mask_durations(k, l);
          }
          sums.push_back(sum);
        }

        double median = nc::NdArray(sums).median().item();
        new_durations[i] += p * median;
      }
    }

    return {durations, new_durations};
  }

  std::vector<std::pair<Model *, std::vector<double>>> compute(std::string &variant_name, Worker *&worker)
  {
    std::vector<std::pair<Model *, std::vector<double>>> results;
    Model *variant;
    for (int batch_size : BATCH_SIZES)
    {
      variant = new Model(*this->load_model_metadata(worker->get_hardware_platform(), variant_name));
      variant->batch_size = batch_size;

      if (worker->percent_occupation(variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY || variant->get_profile_throughput() == 0)
        continue;

      std::vector<Model *> models;
      float perf_drop;

      if (worker->get_total_running_variants() > 0)
      {
        models.push_back(variant);
        models.insert(models.end(), worker->get_variants().begin(), worker->get_variants().end());

        std::string key = this->build_key(worker->get_hardware_platform(), models);

        if (this->history.find(key) != this->history.end())
        {
          perf_drop = this->history[key];
          results.emplace_back(variant, perf_drop);
          continue;
        }

        auto [durations, new_durations] = heuristic_roomie(models);

        if (!nc::all(new_durations >= durations))
        {
          std::ostringstream oss;
          oss << "Bad algorithms for ";
          for (auto *model : models)
          {
            oss << "(" << model->name << ", " << model->batch_size << ") ";
          }
          oss << "\nDurations=" << durations << "\nNew durations=" << new_durations;
          throw std::runtime_error(oss.str());
        }

        perf_drop = ((new_durations - durations) / new_durations).median().item();
        this->history[key] = perf_drop;
      }
      else
      {
        perf_drop = 0.0;
      }

      results.emplace_back(variant, perf_drop);
    }

    return results;
  }
};

#endif // ROOMIE_SCHEDULER_H