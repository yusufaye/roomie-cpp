#ifndef ROOMIE_SCHEDULER_H
#define ROOMIE_SCHEDULER_H

#include <set>
#include <cmath>
#include <math.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

using namespace std;

enum class Strategy
{
  EQUALY,
  FAIRLY,
  FCFS
};

std::vector<std::vector<int>> equaly(const std::vector<std::vector<int>> &resources_required_per_block,
                                     const std::vector<int> &max_blocks,
                                     const std::vector<int> &boundaries)
{
  int N = resources_required_per_block.size(); // Number of kernels
  int M = boundaries.size();                   // Number of resource types

  std::vector<std::vector<int>> maximum_resources_granted(N, std::vector<int>(M, 0));
  std::vector<int> free;
  for (size_t i = 0; i < M; i++)
  {
    free.push_back(boundaries[i]);
  }

  // Equally distribute boundaries
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      maximum_resources_granted[i][j] = boundaries[j] / N;
    }
  }

  // Initial allocation
  for (int i = 0; i < N; ++i)
  {
    int max_block = max_blocks[i];
    for (int j = 0; j < M; ++j)
    {
      if (resources_required_per_block[i][j] > 0)
      {
        int possible = maximum_resources_granted[i][j] / resources_required_per_block[i][j];
        max_block = std::min(max_block, possible);
      }
    }
    for (int j = 0; j < M; ++j)
    {
      free[j] -= resources_required_per_block[i][j] * max_block;
    }
  }

  // Reallocate unused resources
  bool allocated = true;
  while (allocated)
  {
    allocated = false;
    for (int i = 0; i < N; ++i)
    {
      bool can_allocate = true;
      for (int j = 0; j < M; ++j)
      {
        if (free[j] < resources_required_per_block[i][j])
        {
          can_allocate = false;
          break;
        }
      }
      if (can_allocate)
      {
        for (int j = 0; j < M; ++j)
        {
          free[j] -= resources_required_per_block[i][j];
          maximum_resources_granted[i][j] += resources_required_per_block[i][j];
        }
        allocated = true;
      }
    }
  }

  return maximum_resources_granted;
}

std::vector<std::vector<int>> fairly(
    const std::vector<std::vector<int>> &resources_required_per_block,
    const std::vector<int> &max_blocks,
    const std::vector<int> &boundaries)
{
  int N = resources_required_per_block.size(); // Number of kernels
  int M = boundaries.size();                   // Number of resource types

  // Step 1: Sum over columns
  std::vector<int> total_resources_required_per_block(M, 0);
  for (int j = 0; j < M; ++j)
  {
    for (int i = 0; i < N; ++i)
    {
      total_resources_required_per_block[j] += resources_required_per_block[i][j];
    }
  }

  // Step 2: Fair distribution
  std::vector<std::vector<int>> maximum_resources_granted(N, std::vector<int>(M, 0));
  for (int j = 0; j < M; ++j)
  {
    if (total_resources_required_per_block[j] > 0)
    {
      for (int i = 0; i < N; ++i)
      {
        maximum_resources_granted[i][j] = boundaries[j] * resources_required_per_block[i][j] / total_resources_required_per_block[j];
      }
    }
  }

  // Step 3: Initial allocation
  std::vector<int> free = boundaries;
  for (int i = 0; i < N; ++i)
  {
    int max_block = max_blocks[i];
    for (int j = 0; j < M; ++j)
    {
      if (resources_required_per_block[i][j] > 0)
      {
        int possible = maximum_resources_granted[i][j] / resources_required_per_block[i][j];
        max_block = std::min(max_block, possible);
      }
    }
    for (int j = 0; j < M; ++j)
    {
      free[j] -= resources_required_per_block[i][j] * max_block;
    }
  }

  // Step 4: Reallocate unused resources
  bool allocated = true;
  while (allocated)
  {
    allocated = false;
    for (int i = 0; i < N; ++i)
    {
      bool can_allocate = true;
      for (int j = 0; j < M; ++j)
      {
        if (free[j] < resources_required_per_block[i][j])
        {
          can_allocate = false;
          break;
        }
      }
      if (can_allocate)
      {
        for (int j = 0; j < M; ++j)
        {
          free[j] -= resources_required_per_block[i][j];
          maximum_resources_granted[i][j] += resources_required_per_block[i][j];
        }
        allocated = true;
      }
    }
  }

  return maximum_resources_granted;
}

std::vector<std::vector<int>> fcfs(
    const std::vector<std::vector<int>> &resources_required_per_block,
    const std::vector<int> &max_blocks,
    const std::vector<int> &boundaries)
{
  int N = resources_required_per_block.size(); // Number of kernels
  int M = boundaries.size();                   // Number of resource types

  std::vector<std::vector<int>> maximum_resources_granted(N, std::vector<int>(M, 0));

  // Step 1: First-Come-First-Served allocation
  for (int i = 0; i < N; ++i)
  {
    // Compute free resources
    std::vector<int> free(M, 0);
    for (int j = 0; j < M; ++j)
    {
      int used = 0;
      for (int k = 0; k < N; ++k)
      {
        used += maximum_resources_granted[k][j];
      }
      free[j] = boundaries[j] - used;
    }

    // Compute max blocks for this kernel
    int max_block = max_blocks[i];
    for (int j = 0; j < M; ++j)
    {
      if (resources_required_per_block[i][j] > 0)
      {
        int possible = free[j] / resources_required_per_block[i][j];
        max_block = std::min(max_block, possible);
      }
    }

    // Allocate resources
    for (int j = 0; j < M; ++j)
    {
      maximum_resources_granted[i][j] = resources_required_per_block[i][j] * max_block;
    }
  }

  // Step 2: Reallocate unused resources
  bool allocated = true;
  while (allocated)
  {
    allocated = false;

    // Recompute free resources
    std::vector<int> free(M, 0);
    for (int j = 0; j < M; ++j)
    {
      int used = 0;
      for (int k = 0; k < N; ++k)
      {
        used += maximum_resources_granted[k][j];
      }
      free[j] = boundaries[j] - used;
    }

    for (int i = 0; i < N; ++i)
    {
      bool can_allocate = true;
      for (int j = 0; j < M; ++j)
      {
        if (free[j] < resources_required_per_block[i][j])
        {
          can_allocate = false;
          break;
        }
      }

      if (can_allocate)
      {
        for (int j = 0; j < M; ++j)
        {
          free[j] -= resources_required_per_block[i][j];
          maximum_resources_granted[i][j] += resources_required_per_block[i][j];
        }
        allocated = true;
      }
    }
  }

  return maximum_resources_granted;
}

double duration_after_interference(Model *model)
{
  double new_duration = 0;
  for (auto &kernel : model->get_kernels())
  {
    new_duration += kernel->duration_after_interference();
  }
  if (model->initial_duration() > new_duration)
  {
    spdlog::error("Duration after interference should be greater or equal to initial duration (new {} < {})", new_duration, model->initial_duration());
    throw invalid_argument("Kernel interference failed.");
  }
  return new_duration;
}

void reset(Model *model)
{
  for (auto &kernel : model->get_kernels())
  {
    kernel->reset();
  }
}

std::vector<double> simulate_interferance(
    std::vector<Model *> &models,
    const std::vector<int> boundaries,
    Strategy strategy = Strategy::FCFS,
    std::vector<int> starts = {},
    bool one_interference_at_most = true,
    bool interfere_all_kernels = false,
    float prob = 0.2)
{
  auto resource_distribution_strategy = equaly;

  if (strategy == Strategy::EQUALY)
  {
    resource_distribution_strategy = equaly;
  }
  else if (strategy == Strategy::FAIRLY)
  {
    resource_distribution_strategy = fairly;
  }
  else if (strategy == Strategy::FCFS)
  {
    resource_distribution_strategy = fcfs;
  }
  else
  {
    throw invalid_argument("Strategy unrecognized.");
  }

  // Reset kernel interferences.
  for (auto &model : models)
  {
    for (auto &kernel : model->get_kernels())
    {
      kernel->reset();
    }
  }

  std::vector<float> durations;
  for (auto &model : models)
  {
    for (auto &op : model->get_kernels())
    {
      durations.push_back(op->duration);
    }
  }

  float min_duration = max({2 * *min_element(durations.begin(), durations.end()),
                            accumulate(durations.begin(), durations.end(), 0.0f) / durations.size(),
                            median(durations)});

  if (starts.empty())
  {
    starts.resize(models.size(), 0);
  }
  else if (starts.size() != models.size())
  {
    throw invalid_argument("Starts size mismatch with models.");
  }

  std::vector<int> kernel_positions = starts;
  set<int> completed;
  std::vector<std::vector<double>> duration_after_interferences(models.size());
  std::vector<double> new_durations(models.size(), 0.0f);

  while (true)
  {
    vector<Kernel *> kernels;
    for (size_t i = 0; i < models.size(); ++i)
    {
      while (prob > 0 && bernoulli(prob) && kernel_positions[i] < models[i]->get_kernels().size())
      {
        kernel_positions[i]++;
      }
      Kernel *kernel = nullptr;
      if (kernel_positions[i] < models[i]->get_kernels().size())
      {
        kernel = models[i]->get_kernels()[kernel_positions[i]];
      }
      kernels.push_back(kernel);
    }

    vector<int> index_running_models;
    for (size_t i = 0; i < kernels.size(); ++i)
    {
      if (kernels[i] != nullptr)
      {
        index_running_models.push_back(i);
      }
    }

    kernels.erase(remove(kernels.begin(), kernels.end(), nullptr), kernels.end());

    // No more interference, exit the loop.
    if (kernels.size() == 1)
    {
      kernel_positions[index_running_models[0]]++;
      continue;
    }
    else if (kernels.empty())
    {
      break;
    }

    // Decending order.
    std::vector<int> orders;
    for (const auto &kernel : kernels)
    {
      orders.push_back(-kernel->order());
    }
    std::vector<size_t> indeces = argsort(orders);

    reorder_vector(kernel_positions, indeces);
    reorder_vector(index_running_models, indeces);

    std::vector<std::vector<int>> resources_required_per_block;
    std::vector<int> maximum_blocks;
    for (const auto &kernel : kernels)
    {
      resources_required_per_block.push_back(kernel->resource_required_per_block());
      maximum_blocks.push_back(kernel->max_blocks());
    }

    // 2. Distributed the resources over kernels.
    auto maximum_resources_granted = resource_distribution_strategy(resources_required_per_block, maximum_blocks, boundaries);

    // 3. Determine the new maximum number of blocks.
    std::vector<int> new_max_blocks(kernels.size(), 0);
    // std::vector<std::tuple<int, int, int>> resources_required;
    for (int i = 0; i < maximum_resources_granted.size(); ++i)
    {
      // 3.1. Determine the required resource with respect to the maximum block.
      for (size_t j = 0; j < boundaries.size(); j++)
      {
        if (resources_required_per_block[i][j] > 0)
        {
          auto resources_required = min(
              resources_required_per_block[i][j] * maximum_blocks[i],
              maximum_resources_granted[i][j]);

          // Determine the new maximum blocks.
          new_max_blocks[i] = min(
              maximum_blocks[i],
              (int)floor((float)maximum_resources_granted[i][j] / resources_required_per_block[i][j]));
        }
      }
    }

    // Make sure that at least one kernel launches..
    if (all_of(new_max_blocks.begin(), new_max_blocks.end(), [](int x)
               { return x == 0; }))
    {
      new_max_blocks[0] = maximum_blocks[0];
    }

    for (int i = 0; i < kernels.size(); ++i)
    {
      kernels[i]->xxx_max_blocks_granted = new_max_blocks[i];
    }

    std::vector<float> xxx_durations;
    std::vector<std::pair<float, float>> occupancies;
    for (const auto &kernel : kernels)
    {
      xxx_durations.push_back(kernel->xxx_duration);
      occupancies.emplace_back(kernel->achieved_occupancy, kernel->new_occupancy());
    }

    // Filter running kernels and recompute the new duration (or extended duration).
    std::vector<bool> running_kernel_indices;
    for (auto &occ : occupancies)
    {
      running_kernel_indices.push_back(std::get<1>(occ) > 0);
    }

    vector<float> factors;
    for (size_t i = 0; i < occupancies.size(); ++i)
    {
      if (running_kernel_indices[i])
      {
        factors.push_back(std::get<0>(occupancies[i]) / min(std::get<0>(occupancies[i]), std::get<1>(occupancies[i])));
      }
    }

    // Determine the extended new duration during the interference.
    vector<float> extended_durations;
    for (size_t i = 0; i < factors.size(); ++i)
    {
      extended_durations.push_back(factors[i] * xxx_durations[i]);
    }

    int first_kernel_to_finish = argmin(extended_durations);
    float delta = extended_durations[first_kernel_to_finish];

    vector<float> equivalent_work_times;
    for (float f : factors)
    {
      equivalent_work_times.push_back(delta / f);
    }

    vector<float> additional_times(running_kernel_indices.size(), delta);
    for (size_t i = 0, j = 0; i < running_kernel_indices.size(); ++i)
    {
      if (running_kernel_indices[i])
      {
        additional_times[i] = delta - equivalent_work_times[j++];
      }
    }

    for (size_t i = 0; i < kernels.size(); ++i)
    {
      int index_model = index_running_models[i];
      kernels[i]->set_order(kernels[i]->order() + 1);
      if (running_kernel_indices[i])
      {
        kernels[i]->xxx_duration = xxx_durations[i];
        if (std::get<0>(occupancies[i]) > std::get<1>(occupancies[i]))
        {
          kernels[i]->xxx_additional_duration += additional_times[i];
        }
      }
      else
      {
        kernels[i]->xxx_additional_duration += delta;
      }

      if (first_kernel_to_finish == i || (running_kernel_indices[i] && (kernels[i]->xxx_duration < min_duration || one_interference_at_most)))
      {
        kernel_positions[index_model]++;
        if (interfere_all_kernels)
        {
          kernel_positions[index_model] %= models[index_model]->get_kernels().size();
        }
        if (kernel_positions[index_model] == starts[index_model])
        {
          completed.insert(index_model);
          duration_after_interferences[index_model].push_back(duration_after_interference(models[index_model]));
          reset(models[index_model]);
        }
      }
    }

    if (completed.size() == models.size())
      break;
  }

  for (size_t i = 0; i < duration_after_interferences.size(); ++i)
  {
    if (!duration_after_interferences[i].empty())
    {
      new_durations[i] = median(duration_after_interferences[i]);
    }
    else
    {
      new_durations[i] = duration_after_interference(models[i]);
    }
  }

  return new_durations;
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
      spdlog::debug("🤕[Roomie] Warning no variant candidate found");
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

  std::pair<std::vector<double>, std::vector<double>> roomie(std::vector<Model *> &models)
  {
    std::vector<double> durations;
    for (const auto model : models)
    {
      durations.push_back(model->initial_duration());
    }
    NvidiaGpuSpec gpu_spec(models[0]->get_kernels()[0]->capability_major, models[0]->get_kernels()[0]->capability_minor);
    auto boundaries = gpu_spec.boundaries();
    auto new_durations = simulate_interferance(models, boundaries);

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

        auto [durations, new_durations] = roomie(models);

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