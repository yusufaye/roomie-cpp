#ifndef USHER_V1_SCHEDULER_H
#define USHER_V1_SCHEDULER_H

#include <math.h>
#include "usher_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"



class UsherV1Scheduler : public UsherScheduler
{
public:
  UsherV1Scheduler() {}

  std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates, bool scaling=false) override
  {
    auto variant_workers = usher(workers, variant_candidates);

    std::sort(variant_workers.begin(), variant_workers.end(), [](const std::tuple<Model *, Worker *, float> a, const std::tuple<Model *, Worker *, float> b)
              {
                if (std::get<0>(a)->get_throughput() == std::get<0>(b)->get_throughput())
                {
                  return std::get<1>(a)->get_total_running_variants() < std::get<1>(b)->get_total_running_variants();
                }
                return std::get<0>(a)->get_throughput() > std::get<0>(b)->get_throughput();
              });

    if (variant_workers.empty())
    {
      spdlog::debug("🤕[Usher] Warning no variant candidate found");
      return {nullptr, nullptr};
    }
    auto variant = std::get<0>(variant_workers[0]);
    auto worker = std::get<1>(variant_workers[0]);
    return {variant, worker};
  }

  std::vector<std::tuple<Model *, Worker *, float>> usher(std::vector<Worker *> workers, std::vector<string> variant_candidates)
  {
    std::vector<std::tuple<Model *, Worker *, float>> variant_workers;
    for (const auto &variant_name : variant_candidates)
    {
      for (const int &batch_size : BATCH_SIZES)
      {
        auto groups = variant_grouping(workers, variant_name, batch_size);
        auto result = decision_configuration_and_placement(groups, workers);
        for (const auto &item : result)
        {
          variant_workers.push_back(item);
        }
      }
    }
    return variant_workers;
  }

  std::vector<std::vector<UsherModel *>> variant_grouping(vector<Worker *> workers, const string &variant_name, int batch_size, int max_variants_group = 4)
  {
    std::vector<UsherModel *> variants;
    std::vector<std::string> hardware_platforms;

    for (Worker *worker : workers)
    {
      if (std::find(hardware_platforms.begin(), hardware_platforms.end(), worker->get_hardware_platform()) != hardware_platforms.end())
      {
        continue;
      }
      hardware_platforms.push_back(worker->get_hardware_platform());
      Model *variant = new Model(*this->load_model_metadata(worker->get_hardware_platform(), variant_name));
      variant->batch_size = batch_size;
      if (variant->get_throughput() == 0)
      {
        continue;
      }
      UsherModel *usher_variant = new UsherModel(variant, worker);
      variants.push_back(usher_variant);
    }

    std::vector<std::vector<UsherModel *>> groups;
    for (auto &worker : workers)
    {
      std::vector<UsherModel *> group;
      for (auto &variant : worker->get_variants())
      {
        group.push_back(new UsherModel(variant, worker));
      }
      groups.push_back(group);
    }
    for (auto &variant : variants)
    {
      groups.push_back({variant});
    }

    int MAX_GROUPS = workers.size();
    /* For Jetson only */
    int MIN_GROUPS = 1;
    for (const auto &worker : workers)
    {
      if (worker->get_total_running_variants() > 0)
      {
        MIN_GROUPS++;
      }
    }
    MIN_GROUPS = std::min(MAX_GROUPS, MIN_GROUPS);

    // while ((groups.size() > MIN_GROUPS) && (groups.size() > MAX_GROUPS) && (groups[0].size() < max_variants_group))
    while ((groups.size() > MIN_GROUPS) && (groups.size() > MAX_GROUPS))
    {
      std::vector<double> c_reqs;
      std::vector<double> m_reqs;
      for (auto &group : groups)
      {
        double c_req = 0;
        double m_req = 0;
        for (auto &variant : group)
        {
          c_req += variant->c_req;
          m_req += variant->m_req;
        }
        c_reqs.push_back(c_req);
        m_reqs.push_back(m_req);
      }

      std::vector<std::tuple<int, int, double>> D;
      for (int i = 0; i < groups.size(); ++i)
      {
        for (int j = i + 1; j < groups.size(); ++j)
        {
          double distance = std::abs((c_reqs[i] + c_reqs[j]) - (m_reqs[i] + m_reqs[j]));
          D.push_back(std::make_tuple(i, j, distance));
        }
      }

      std::vector<std::vector<UsherModel *>> new_groups;
      std::vector<int> skip_indices;
      int N = groups.size() / 2;
      for (int i = 0; i < N; ++i)
      {
        auto min_distance = *std::min_element(D.begin(), D.end(), [](const std::tuple<int, int, double> &a, const std::tuple<int, int, double> &b)
                                              { return std::get<2>(a) < std::get<2>(b); });
        skip_indices.push_back(std::get<0>(min_distance));
        skip_indices.push_back(std::get<1>(min_distance));
        new_groups.push_back(groups[std::get<0>(min_distance)]);
        new_groups.back().insert(new_groups.back().end(), groups[std::get<1>(min_distance)].begin(), groups[std::get<1>(min_distance)].end());
        D.erase(std::remove_if(D.begin(), D.end(), [&](const std::tuple<int, int, double> &d)
                               { return std::find(skip_indices.begin(), skip_indices.end(), std::get<0>(d)) != skip_indices.end() ||
                                        std::find(skip_indices.begin(), skip_indices.end(), std::get<1>(d)) != skip_indices.end(); }),
                D.end());
      }

      for (int i = 0; i < groups.size(); ++i)
      {
        if (std::find(skip_indices.begin(), skip_indices.end(), i) == skip_indices.end())
        {
          new_groups.push_back(groups[i]);
        }
      }

      groups = new_groups;
    }

    spdlog::debug("---> Min={}, Max={}, Group-size={}", MIN_GROUPS, MAX_GROUPS, groups.size());

    return groups;
  }

  std::vector<std::tuple<Model *, Worker *, float>> decision_configuration_and_placement(std::vector<std::vector<UsherModel *>> groups, std::vector<Worker *> workers)
  {
    std::vector<std::tuple<Model *, Worker *, float>> variant_worker;
    for (auto &group : groups)
    {
      if (group.empty())
      {
        continue;
      }
      // 1. Generate possible configurations, i.e., list of (batch size, replica degree) for each variant in a group (Gi).
      // we will consider using only one replica at time and then scale accordingly.
      // 2. For each configuration, find the placement to minimize the cost or maximize the goodput.
      std::vector<std::tuple<Model *, Worker *, float>> placements = placement(group, workers);
      for (const auto &placement : placements)
      {
        variant_worker.push_back(placement);
      }
    }
    return variant_worker;
  }

  std::vector<std::tuple<Model *, Worker *, float>> placement(const std::vector<UsherModel *> &group, const std::vector<Worker *> &workers)
  {
    std::vector<Worker *> GiGPU;
    for (const auto &worker : workers)
    {
      if (worker->get_total_running_variants() == 0)
      {
        for (const auto &item: group)
        {
          if (item->model->id == 0)
          {
            return {{item->model, worker, 0.0}};
          }
        }
      }
    }
    for (const auto &wrapper : group)
    {
      if (wrapper->model->id > 0)
      {
        for (const auto &worker : workers)
        {
          if (worker->exists(wrapper->model->id) && std::find(GiGPU.begin(), GiGPU.end(), worker) == GiGPU.end())
          {
            GiGPU.push_back(worker);
          }
        }
      }
    }

    std::vector<UsherModel *> variants = group;
    std::vector<UsherModel *> c_heavy_variants;
    std::vector<UsherModel *> m_heavy_variants;
    std::vector<UsherModel *> light_variants;

    for (const auto &variant : variants)
    {
      if (Cheavy(*variant))
      {
        c_heavy_variants.push_back(variant);
      }
      else if (Mheavy(*variant))
      {
        m_heavy_variants.push_back(variant);
      }
      else
      {
        light_variants.push_back(variant);
      }
    }

    std::sort(c_heavy_variants.begin(), c_heavy_variants.end(), [](const UsherModel *a, const UsherModel *b)
              { return a->c_req + a->m_req > b->c_req + b->m_req; });
    std::sort(m_heavy_variants.begin(), m_heavy_variants.end(), [](const UsherModel *a, const UsherModel *b)
              { return a->c_req + a->m_req > b->c_req + b->m_req; });

    std::vector<std::pair<UsherModel *, UsherModel *>> final_model_list;
    while (!c_heavy_variants.empty())
    {
      if (m_heavy_variants.empty())
      {
        break;
      }
      UsherModel *c_heavy_variant = c_heavy_variants[0];
      c_heavy_variants.erase(c_heavy_variants.begin());
      UsherModel *m_heavy_variant = m_heavy_variants[0];
      m_heavy_variants.erase(m_heavy_variants.begin());
      final_model_list.push_back({c_heavy_variant, m_heavy_variant});
    }

    std::vector<UsherModel *> remaining_variants = c_heavy_variants;
    remaining_variants.insert(remaining_variants.end(), m_heavy_variants.begin(), m_heavy_variants.end());
    remaining_variants.insert(remaining_variants.end(), light_variants.begin(), light_variants.end());

    while (!remaining_variants.empty())
    {
      if (remaining_variants.size() == 1)
      {
        final_model_list.push_back(std::make_pair(remaining_variants[0], remaining_variants[0]));
        remaining_variants.erase(remaining_variants.begin());
      }
      else
      {
        final_model_list.push_back(std::make_pair(remaining_variants[0], remaining_variants[1]));
        remaining_variants.erase(remaining_variants.begin(), remaining_variants.begin() + 2);
      }
    }

    std::vector<std::tuple<Model *, Worker *, float>> variant_workers;
    std::vector<Worker *> worker_candidates; /* For Jetson only. */
    // for (auto &worker : workers)
    // {
    //   if (worker->get_total_running_variants() == 0 &&
    //       std::find(worker_candidates.begin(), worker_candidates.end(), worker) == worker_candidates.end())
    //   {
    //     worker_candidates.push_back(worker);
    //   }
    // }
    for (auto &wrapper_pair : final_model_list)
    {
      worker_candidates.clear();
      UsherModel *running = nullptr;
      UsherModel *deploying = nullptr;
      for (auto &wrapper : {wrapper_pair.first, wrapper_pair.second})
      {
        if (wrapper->model->id > 0)
        {
          running = wrapper;
        }
        else
        {
          deploying = wrapper;
        }
      }
      if (deploying == nullptr)
      {
        continue;
      }
      if (running != nullptr)
      {
        // Find the runner's worker.
        for (auto &worker : workers)
        {
          if (worker->exists(running->model->id) && worker->percent_occupation(deploying->model->get_memory()) <= MAX_GPU_MEMORY_OCCUPANCY)
          {
            worker_candidates.push_back(worker);
            break;
          }
        }
      }

      /* The placement algorithm tries to assign one group of models Gi to the same GPU set to maximize Cuti and Muti. Therefore, to assign the models in Gi , the algorithm prioritizes the GPUs that are already assigned with the models in Gi , and initializes a new GPU only when no used GPU can host a model. */
      if (worker_candidates.empty())
      {
        for (auto &worker : workers)
        {
          if (worker->get_total_running_variants() == 0 && worker->percent_occupation(deploying->model->get_memory()) <= MAX_GPU_MEMORY_OCCUPANCY) // empty worker
          {
            worker_candidates.push_back(worker);
          }
        }
      }

      if (worker_candidates.empty() && !GiGPU.empty())
      {
        for (auto &worker : GiGPU)
        {
          if (worker->percent_occupation(deploying->model->get_memory()) <= MAX_GPU_MEMORY_OCCUPANCY)
          {
            worker_candidates.push_back(worker);
          }
        }
      }

      std::sort(worker_candidates.begin(), worker_candidates.end(),
                [](Worker *a, Worker *b)
                {
                  return a->get_free_memory() > b->get_free_memory();
                });

      if (worker_candidates.empty())
      {
        continue;
      }

      std::vector<float> c_space_m_space;
      UsherModel *tmp;
      for (const auto &worker : worker_candidates)
      {
        float total = 0;
        for (auto &v : worker->get_variants())
        {
          tmp = new UsherModel(v, worker);
          total += tmp->c_req + tmp->m_req;
        }
        total += deploying->c_req + deploying->m_req;
        c_space_m_space.push_back(total);
      }

      int argmax_v = std::distance(c_space_m_space.begin(),
                                   std::max_element(c_space_m_space.begin(), c_space_m_space.end()));
      auto selected_worker = worker_candidates[argmax_v];

      if (std::find(GiGPU.begin(), GiGPU.end(), selected_worker) == GiGPU.end())
      {
        GiGPU.push_back(selected_worker);
      }

      variant_workers.push_back({deploying->model, selected_worker, c_space_m_space[argmax_v]});
    }
    return variant_workers;
  }
};

#endif // USHER_V1_SCHEDULER_H