#ifndef USHER_SCHEDULER_H
#define USHER_SCHEDULER_H

#include <math.h>
#include "NumCpp.hpp"
#include "base_scheduler.h"
#include "utils/general.h"
#include "utils/datastore.h"

class ModelWrapper {
  public:
  Model* model;
  float Creq;
  float Mreq;
  ModelWrapper(Model* model_): model(model_) { }
};

float Mreq(Model &variant, float total_memory)
{
  // """Mreq of a model is the HIGHEST PERCENTAGE of the total memory space of a GPU consumed by the model at any point during its execution.

  // Args:
  //   variant (Model): _description_

  // Returns:
  //   float: the highest percentage of memory
  // """
  return variant.get_memory() / total_memory * 100;
}

float Creq(Model &variant)
{
  // """Creq of a model is the HIGHEST PERCENTAGE of the total computation space of a GPU consumed by the model at any point during its execution.

  // Args:
  //   variant (Model): _description_

  // Returns:
  //   float: the highest percentage of computation
  // """
  std::vector<float> achieved_occupancy;
  for (auto &it : variant.get_kernels())
    achieved_occupancy.push_back(it->achieved_occupancy);
  return nc::mean(nc::NdArray(achieved_occupancy)).item();
}

class UsherSchaduling : public Scheduler
{
public:
  UsherSchaduling() {}

  pair<Model *, Worker *> schedule(vector<Worker *> workers, vector<string> variant_candidates) override
  {
    auto variant_workers = usher(workers, variant_candidates);
    sort(variant_workers.begin(), variant_workers.end(), [](const auto &a, const auto &b)
         { return make_pair(-a[2], -a[0]->profile_throughput) < make_pair(-b[2], -b[0]->profile_throughput); });
    if (variant_workers.empty())
    {
      return {nullptr, nullptr};
    }
    auto variant = variant_workers[0][0];
    auto worker = variant_workers[0][1];
    return {variant, worker};
  }

  vector<tuple<Model *, Worker *, float>> usher(vector<Worker *> workers, vector<string> variant_candidates)
  {
    vector<tuple<Model *, Worker *, float>> variant_workers;
    for (const auto &variant_name : variant_candidates)
    {
      for (const auto &batch_size : BATCH_SIZES)
      {
        auto groups = variant_grouping(workers, variant_name, batch_size);
        variant_workers.insert(variant_workers.end(), decision_configuration_and_placement(groups, workers).begin(), decision_configuration_and_placement(groups, workers).end());
      }
    }
    return variant_workers;
  }

  std::vector<std::vector<Model *>> variant_grouping(vector<Worker *> workers, const string &variant_name, int batch_size, int max_variants_group = 4)
  {
    std::vector<Model *> variants;
    std::vector<std::string> hardware_platforms;

    for (Worker *worker : workers)
    {
      if (std::find(hardware_platforms.begin(), hardware_platforms.end(), worker->get_hardware_platform()) != hardware_platforms.end())
      {
        continue;
      }
      hardware_platforms.push_back(worker->get_hardware_platform());
      Model *variant = get_model(worker->get_hardware_platform(), worker->get_hardware_platform());
      variant->batch_size = batch_size;
      if (variant->get_profile_throughput() == 0)
      {
        continue;
      }
      // variant->Mreq = Mreq(variant, worker.total_memory);
      variants.push_back(variant);
    }

    std::vector<std::vector<Model *>> groups;
    for (auto &worker : workers)
    {
      if (!worker.variants.empty())
      {
        groups.push_back(worker.variants);
      }
    }
    for (auto &variant : variants)
    {
      groups.push_back({variant});
    }

    int MAX_GROUPS = workers.size();
    int MIN_GROUPS = std::min(MAX_GROUPS, 2);

    while (groups.size() > MIN_GROUPS && groups.size() > MAX_GROUPS && groups[0].size() < max_variants_group)
    {
      std::vector<double> c_reqs;
      std::vector<double> m_reqs;
      for (auto &group : groups)
      {
        double c_req = 0;
        double m_req = 0;
        for (auto &variant : group)
        {
          c_req += Creq(*variant);
          m_req += Mreq(*variant);
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

      std::vector<std::vector<Model *>> new_groups;
      std::vector<int> skip_indices;
      int N = groups.size() / 2;
      for (int i = 0; i < N; ++i)
      {
        auto min_distance = *std::min_element(D.begin(), D.end(), [](const auto &a, const auto &b)
                                              { return std::get<2>(a) < std::get<2>(b); });
        skip_indices.push_back(std::get<0>(min_distance));
        skip_indices.push_back(std::get<1>(min_distance));
        new_groups.push_back(groups[std::get<0>(min_distance)]);
        new_groups.back().insert(new_groups.back().end(), groups[std::get<1>(min_distance)].begin(), groups[std::get<1>(min_distance)].end());
        D.erase(std::remove_if(D.begin(), D.end(), [&](const auto &d)
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

    return groups;
  }

  std::vector<std::tuple<Model*, Worker*, float>> decision_configuration_and_placement(std::vector<std::vector<Model*>> groups, std::vector<Worker*> workers)
  {
    std::vector<std::tuple<Model, Worker, float>> variant_worker;
    for (auto &group : groups)
    {
      // 1. Generate possible configurations, i.e., list of (batch size, replica degree) for each variant in a group (Gi).
      // we will consider using only one replica at time and then scale accordingly.
      // 2. For each configuration, find the placement to minimize the cost or maximize the goodput.
      std::vector<std::tuple<Model, Worker, float>> placements = placement(group, workers);
      variant_worker.insert(variant_worker.end(), placements.begin(), placements.end());
    }
    return variant_worker;
  }


  std::vector<std::tuple<Model*, Worker*, float>> placement(const std::vector<Model*>& group, const std::vector<Worker*>& workers) {
    std::vector<Worker*> GiGPU;
    for (const auto& variant : group) {
        if (variant ->id) {
            for (const auto& worker : workers) {
                if (std::find(worker->get_variants().begin(), worker->get_variants().end(), variant) != worker->get_variants().end() && std::find(GiGPU.begin(), GiGPU.end(), worker) == GiGPU.end()) {
                    GiGPU.push_back(worker);
                }
            }
        }
    }

    std::vector<Model*> variants = group;
    std::vector<Model*> c_heavy_variants;
    std::vector<Model*> m_heavy_variants;
    std::vector<Model*> light_variants;

    for (const auto& variant : variants) {
        if (Cheavy(variant)) {
            c_heavy_variants.push_back(variant);
        } else if (Mheavy(variant)) {
            m_heavy_variants.push_back(variant);
        } else {
            light_variants.push_back(variant);
        }
    }

    std::sort(c_heavy_variants.begin(), c_heavy_variants.end(), [](const Model& a, const Model& b) { return a.Creq + a.Mreq > b.Creq + b.Mreq; });
    std::sort(m_heavy_variants.begin(), m_heavy_variants.end(), [](const Model& a, const Model& b) { return a.Creq + a.Mreq > b.Creq + b.Mreq; });

    std::vector<std::pair<Model, Model>> final_model_list;
    while (!c_heavy_variants.empty()) {
        if (m_heavy_variants.empty()) {
            break;
        }
        Model c_heavy_variant = c_heavy_variants[0];
        c_heavy_variants.erase(c_heavy_variants.begin());
        Model m_heavy_variant = m_heavy_variants[0];
        m_heavy_variants.erase(m_heavy_variants.begin());
        final_model_list.push_back(std::make_pair(c_heavy_variant, m_heavy_variant));
    }

    std::vector<Model> remaining_variants = c_heavy_variants;
    remaining_variants.insert(remaining_variants.end(), m_heavy_variants.begin(), m_heavy_variants.end());
    remaining_variants.insert(remaining_variants.end(), light_variants.begin(), light_variants.end());

    while (!remaining_variants.empty()) {
        if (remaining_variants.size() == 1) {
            final_model_list.push_back(std::make_pair(remaining_variants[0], remaining_variants[0]));
            remaining_variants.erase(remaining_variants.begin());
        } else {
            final_model_list.push_back(std::make_pair(remaining_variants[0], remaining_variants[1]));
            remaining_variants.erase(remaining_variants.begin(), remaining_variants.begin() + 2);
        }
    }
  }

    // std::vector<std::tuple<Model, Worker, float>> variant_worker;
    // for (const auto& variant_pair : final_model_list) {
    //     std::vector<Worker> worker_candidates;
    //     for (const auto& variant : {variant_pair.first, variant_pair.second}) {
    //         if (variant.id) {
    //             for (const auto& worker : workers) {
    //                 if (std::find(worker.variants.begin(), worker.variants.end(), variant) != worker.variants.end()) {
    //                     worker_candidates.push_back(worker);
    //                 }
    //             }
    //         }
    //     }

    //     for (const auto& variant : {variant_pair.first, variant_pair.second}) {
    //         if (variant.id) {
    //             continue;
    //         }
    //         worker_candidates.erase(std::remove_if(worker_candidates.begin(), worker_candidates.end(), [&](const Worker& worker) { return worker.percent_occupation(variant.memory) > 100; }), worker_candidates.end());
    //         if (worker_candidates.empty() && !GiGPU.empty())
};

#endif // USHER_SCHEDULER_H