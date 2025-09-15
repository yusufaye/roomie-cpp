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
  std::vector<int> free = boundaries;

  // Equally distribute boundaries
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      maximum_resources_granted[i][j] = boundaries[j] / N;
    }
  }

  // Initial allocation
  auto max_blocks_granted = max_blocks;
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      if (resources_required_per_block[i][j] > 0)
      {
        max_blocks_granted[i] = std::min(max_blocks_granted[i], maximum_resources_granted[i][j] / resources_required_per_block[i][j]);
      }
    }
    for (int j = 0; j < M; ++j)
    {
      free[j] -= resources_required_per_block[i][j] * max_blocks_granted[i];
    }
  }

  // Reallocate unused resources
  bool allocated, can_allocate = true;
  int i = 0;
  while (allocated)
  {
    if (i++ == 5)
    {
      break;
    }
    spdlog::debug("Free resources: {}", vec2str(free));
    for (int i = 0; i < N; ++i)
    {
      spdlog::debug("\t{}", vec2str(resources_required_per_block[i]));
    }
    allocated = false;
    for (int i = 0; i < N; ++i)
    {
      can_allocate = true;
      /* check if all resources match. */
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
  double new_duration = 0.0;
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
    Strategy strategy = Strategy::EQUALY,
    std::vector<int> starts = {},
    bool one_interference_at_most = true,
    bool interfere_all_kernels = false,
    float prob = 0.2)
{
  if (models.size() == 1)
  {
    return {models[0]->initial_duration()};
  }

  auto resource_distribution_strategy = equaly;

  if (strategy == Strategy::FAIRLY)
  {
    resource_distribution_strategy = fairly;
  }
  else if (strategy == Strategy::FCFS)
  {
    resource_distribution_strategy = fcfs;
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
    for (auto &kernel : model->get_kernels())
    {
      durations.push_back(kernel->duration);
    }
  }

  std::vector<float> tmp = {2 * minimum(durations), mean(durations), median(durations)};
  float min_duration = maximum(tmp);

  if (starts.empty())
  {
    starts.resize(models.size(), 0);
  }
  else if (starts.size() != models.size())
  {
    throw invalid_argument("Starts size mismatch with models.");
  }

  std::vector<int> per_model_kernel_pos = starts;
  std::set<int> completed;
  std::vector<std::vector<double>> duration_after_interferences(models.size());
  std::vector<double> new_durations(models.size(), 0.0f);
  std::vector<Kernel *> kernels;

  while (true)
  {
    kernels.clear();
    for (size_t i = 0; i < models.size(); ++i)
    {
      while (prob > 0 && bernoulli(prob) && per_model_kernel_pos[i] < models[i]->get_kernels().size())
      {
        per_model_kernel_pos[i]++;
      }
      Kernel *kernel = nullptr;
      if (per_model_kernel_pos[i] < models[i]->get_kernels().size())
      {
        kernel = models[i]->get_kernels()[per_model_kernel_pos[i]];
      }
      kernels.push_back(kernel);
    }

    std::vector<int> concurrent_kernel_index;
    for (size_t i = 0; i < kernels.size(); ++i)
    {
      if (kernels[i] != nullptr)
      {
        concurrent_kernel_index.push_back(i);
      }
    }

    kernels.erase(remove(kernels.begin(), kernels.end(), nullptr), kernels.end());

    // No more interference, exit the loop.
    if (kernels.size() == 1)
    {
      per_model_kernel_pos[concurrent_kernel_index[0]]++;
      continue;
    }
    else if (kernels.empty())
    {
      break;
    }

    std::vector<int> orders;
    for (const auto &kernel : kernels)
    {
      orders.push_back(-kernel->order()); // Decending order.
    }
    std::vector<size_t> indeces = argsort(orders);
    reorder_vector(kernels, indeces);
    reorder_vector(per_model_kernel_pos, indeces);
    reorder_vector(concurrent_kernel_index, indeces);

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

    // Make sure that at least one kernel launches.
    if (all_of(new_max_blocks.begin(), new_max_blocks.end(), [](int new_max_blocks)
               { return new_max_blocks == 0; }))
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

    std::vector<float> factors;
    for (size_t i = 0; i < occupancies.size(); ++i)
    {
      if (running_kernel_indices[i])
      {
        factors.push_back(std::get<0>(occupancies[i]) / min(std::get<0>(occupancies[i]), std::get<1>(occupancies[i])));
      }
    }

    // Determine the extended new duration during the interference.
    std::vector<float> extended_durations;
    for (size_t i = 0; i < factors.size(); ++i)
    {
      extended_durations.push_back(factors[i] * xxx_durations[i]);
    }

    if (extended_durations.empty())
    {
      auto max_blocks = new_max_blocks;
      for (size_t i = 0; i < new_max_blocks.size(); i++)
      {
        std::cout << kernels[i] << std::endl;
        spdlog::debug("New occupancy:\n\txxx_max_blocks_granted: {}\n\twarps_per_block: {}\n\twarps_per_multiprocessor: {}\n\toccupancy: {}",
                     kernels[i]->xxx_max_blocks_granted,
                     kernels[i]->get_perf().warpsPerBlock,
                     kernels[i]->get_perf().warpsPerMultiprocessor,
                     (float)kernels[i]->xxx_max_blocks_granted * kernels[i]->get_perf().warpsPerBlock / kernels[i]->get_perf().warpsPerMultiprocessor);
      }
      spdlog::error("====== No running kernel: {} | max_blocks: {} | new_max_blocks: {}",
                    vec2str(running_kernel_indices),
                    vec2str(maximum_blocks),
                    vec2str(new_max_blocks));
    }

    int first_kernel_to_finish = argmin(extended_durations);
    float delta = extended_durations[first_kernel_to_finish];

    std::vector<float> equivalent_work_times;
    for (float factor : factors)
    {
      equivalent_work_times.push_back(delta / factor);
    }

    std::vector<float> additional_times(running_kernel_indices.size(), delta);
    for (size_t i = 0, j = 0; i < running_kernel_indices.size(); ++i)
    {
      if (running_kernel_indices[i])
      {
        additional_times[i] = delta - equivalent_work_times[j++];
      }
    }

    for (size_t i = 0; i < kernels.size(); ++i)
    {
      int index_model = concurrent_kernel_index[i];
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
        per_model_kernel_pos[index_model]++;
        if (interfere_all_kernels)
        {
          per_model_kernel_pos[index_model] %= models[index_model]->get_kernels().size();
        }
        if (per_model_kernel_pos[index_model] == starts[index_model])
        {
          completed.insert(index_model);
          duration_after_interferences[index_model].push_back(duration_after_interference(models[index_model]));
          reset(models[index_model]);
        }
      }
    }

    if (completed.size() == models.size())
    {
      break;
    }
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

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates, bool scaling=false) override
  {
    spdlog::debug("------------ {} -----------", variant_candidates[0]);
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
        if (*it <= 0.5)
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
    int i = 0;
    for (const auto &el : simulations)
    {
      if (i++ > 5)
      {
        break;
      }
      std::vector<std::string> names;
      for (const auto &item : std::get<1>(el)->get_variants())
      {
        names.push_back(item->name);
      }
      spdlog::debug("Variant({}, thr={}) ---> Worker({}, variants={}) | {}",
                    std::get<0>(el)->name, std::get<0>(el)->get_throughput() - std::get<0>(el)->get_throughput() * std::get<2>(el)[0],
                    std::get<1>(el)->get_id(), vec2str(names),
                    vec2str(std::get<2>(el)));
    }

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

  std::pair<std::vector<double>, std::vector<double>> roomie(std::vector<Model *> &models, Worker *worker)
  {
    std::vector<double> durations;
    for (const auto model : models)
    {
      durations.push_back(model->initial_duration());
    }
    auto new_durations = simulate_interferance(models, worker->get_gpu_spec()->boundaries());

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
      for (size_t i = 0; i < variant->get_kernels().size(); i++)
      {
        /* we copy to make sure that two instances don't reference the same kernels vector. */
        variant->get_kernels()[i] = new Kernel(*variant->get_kernels()[i]);
      }
      for (auto &kernel : variant->get_kernels())
      {
        auto perf = worker->get_gpu_spec()->theoretical_occupancy(kernel->thread_block(), kernel->register_per_thread, kernel->shared_memory);
        kernel->set_perf(perf);
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
        auto [durations, new_durations] = roomie(models, worker);

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