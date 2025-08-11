#ifndef AUTO_SCALER_H
#define AUTO_SCALER_H

#include <map>
#include "utils/general.h"
#include "utils/datastore.h"
#include "networking/message.h"
#include "scheduling/base_scheduler.h"

class AutoScaler
{
private:
  Event event_;
  Scheduler *scheduler_;
  DataStore *datastore_;
  std::function<void(const std::string &app_id, Model &variant, Worker &worker)> on_deploy_;
  std::function<void(const std::string &app_id, Model &variant, Worker &worker)> on_stop_;

  int interval = 2; // seconds
  // double threshold = 1.0;
  double threshold = 1.5;
  std::map<string, int> locker_;

public:
  AutoScaler(Scheduler *sched, DataStore *ds, std::function<void(const std::string &app_id, Model &variant, Worker &worker)> on_deploy, std::function<void(const std::string &app_id, Model &variant, Worker &worker)> on_stop, std::function<void(void)> on_update = nullptr)
      : scheduler_(sched), datastore_(ds), on_deploy_(on_deploy), on_stop_(on_stop) {}

  void set_event()
  {
    event_.set();
  }

  void run()
  {
    event_.wait();
    spdlog::info("😎 [auto-scaler] About to start the auto-scaling.");

    // Start monitoring loop
    while (true)
    {
      std::this_thread::sleep_for(std::chrono::seconds(interval));
      std::vector<std::pair<std::string, float>> overloaded;
      for (const auto &[app_id, names] : datastore_->get_registration())
      {
        if (locker_.find(app_id) != locker_.end() && locker_[app_id] > 0)
        {
          // spdlog::warn("Locker for app {} is {}", app_id, locker_[app_id]);
          locker_[app_id]--;
          continue;
        }

        std::vector<Model *> running_variants;
        for (Worker *worker : datastore_->get_workers())
        {
          for (Model *variant : worker->get_variants())
          {
            if (std::find(names.begin(), names.end(), variant->name) != names.end())
            {
              running_variants.push_back(variant);
            }
          }
        }

        if (running_variants.empty())
          continue;

        double throughput = 0.0;
        double workload = 0.0;
        for (Model *variant : running_variants)
        {
          throughput += variant->compute_throughput();
          workload += variant->compute_workload();
        }
        overloaded.push_back({app_id, workload / throughput});
      }
      std::sort(overloaded.begin(), overloaded.end(), [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b)
                { return a.second < b.second; });
      std::string summary = "";
      for (const auto [app_id, ratio] : overloaded)
      {
        summary += "\n\t| " + app_id + ": Ratio=" + std::to_string(ratio);
      }
      spdlog::debug("🔵 [auto-scaler] Summary: {}", summary);

      for (size_t i = 0; i < overloaded.size(); i++)
      {
        downscale(overloaded[i].first, overloaded[i].second);
      }
      for (int i = overloaded.size() - 1; i >= 0; i--)
      {
        if (upscale(overloaded[i].first, overloaded[i].second))
        {
          break;
        }
      }
    }
  }

  bool upscale(const string &app_id, float ratio)
  {
    if (ratio <= threshold)
    {
      return false;
    }
    std::vector<Worker *> _workers;
    for (const auto worker : datastore_->get_workers())
    {
      // if (worker->get_state() == State::SET)
      // {
      //   _workers.push_back(worker);
      // }
      _workers.push_back(worker);
    }
    if (_workers.empty())
    {
      spdlog::warn("⚠️ [auto-scaler] All workers are deploying.");
      return false;
    }
    std::vector<std::string> names;
    for (const auto name : datastore_->get_registered(app_id))
    {
      names.push_back(name);
    }
    auto [variant, worker] = scheduler_->schedule(_workers, names);
    if (variant != nullptr)
    {
      if (worker->percent_occupation(variant->get_memory()) > MAX_GPU_MEMORY_OCCUPANCY)
      {
        throw std::runtime_error("⛔️[auto-scaler] Not enough memory left for " + variant->to_string() + " at " + worker->to_string() + "\n\t| New occupancy: " + std::to_string(worker->percent_occupation(variant->get_memory())) + " (%)");
      }
      on_deploy_(app_id, *variant, *worker);
      locker_[app_id] = 5;
      spdlog::debug("⬆️[auto-scaler] About to perform upscaling:\n\t| {}\n\t| to {}", variant->to_string(), worker->to_string());
      return true;
    }
    return false;
  }

  bool downscale(const string &app_id, float ratio)
  {
    if (ratio < 0.5)
    {
      auto [variant, worker] = downscaling_(app_id, true);
      if (variant != nullptr)
      {
        on_stop_(app_id, *variant, *worker);
        spdlog::debug("⬇️[auto-scaler] About to perform downscaling:\n\t| {}\n\t| to {}", variant->to_string(), worker->to_string());
        return true;
      }
    }
    else if (ratio < 0.8)
    {
      auto [variant, worker] = downscaling_(app_id, false);
      if (variant != nullptr)
      {
        on_stop_(app_id, *variant, *worker);
        spdlog::debug("⬇️[auto-scaler] About to perform downscaling:\n\t| {}\n\t| to {}", variant->to_string(), worker->to_string());
        return true;
      }
    }
    return false;
  }

  std::pair<Model *, Worker *> downscaling_(const string &app_id, bool force)
  {
    std::vector<std::pair<Model *, Worker *>> candidates = datastore_->get_variant_workers(app_id);
    if (candidates.size() > 1)
    {
      if (force)
      {
        std::sort(candidates.begin(), candidates.end(),
                  [](std::pair<Model *, Worker *> &a, std::pair<Model *, Worker *> &b)
                  {
                    return a.first->get_achieved_throughput() < b.first->get_achieved_throughput();
                  });
        return candidates.front();
      }
      else
      {
        double throughput = 0, workload = 0;
        for (const auto &[variant, _] : candidates)
        {
          throughput += variant->compute_throughput();
          workload += variant->compute_workload();
        }

        std::vector<std::pair<Model *, Worker *>> filtered;
        for (const auto &[variant, worker] : candidates)
        {
          double new_throughput = throughput - variant->compute_throughput();
          if ((workload / new_throughput) < threshold)
          {
            filtered.emplace_back(variant, worker);
          }
        }

        if (!filtered.empty())
        {
          std::sort(filtered.begin(), filtered.end(),
                    [](std::pair<Model *, Worker *> &a, std::pair<Model *, Worker *> &b)
                    {
                      return a.second->get_total_running_variants() <
                             b.second->get_total_running_variants();
                    });
          return filtered.front();
        }
      }
    }
    return {nullptr, nullptr};
  }
};

#endif // AUTO_SCALER_H