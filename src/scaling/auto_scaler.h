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
  double threshold = 1.0;
  // double threshold = 1.5;
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
    std::cout << "😎[AUTO-SCALER] About to start the auto-scaling." << std::endl;

    // Start monitoring loop
    while (true)
    {
      std::this_thread::sleep_for(std::chrono::seconds(interval));
      std::pair<std::string, double> most_overloaded_app = {"", 0.0};

      for (const auto &[app_id, names] : datastore_->get_registration())
      {
        if (locker_.find(app_id) != locker_.end() && locker_[app_id] > 0)
        {
          cout << "Locker for app " << app_id << " is " << locker_[app_id] << endl;
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
        double workload   = 0.0;
        for (Model *variant : running_variants)
        {
          throughput  += variant->compute_throughput();
          workload    += variant->compute_workload();
        }
        double ratio = workload / throughput;
        if (ratio > most_overloaded_app.second)
        {
          most_overloaded_app = {app_id, ratio};
        }

        std::cout << "🔵[AUTO-SCALER] For Load=" + std::to_string(workload) + ", Thr=" + std::to_string(throughput) + ", Ratio=" + std::to_string(ratio) << std::endl;
      }

      auto [app_id, ratio] = most_overloaded_app;
      try
      {
        if (ratio > 0)
        {
          auto_scale(app_id, ratio);
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << e.what() << '\n';
      }
    }
  }

  bool auto_scale(const string &app_id, double ratio)
  {
    if (ratio < 0.5)
    {
      auto departure = Downscaling(app_id, true);
      if (departure.first != nullptr)
      {
        cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
        for (const auto variant: datastore_->get_variants(app_id))
        {
          std::cout << "----> " << variant->to_string() << endl;
        }
        
        on_stop_(app_id, *departure.first, *departure.second);
        return true;
      }
    }
    else if (ratio < 0.8)
    {
      auto departure = Downscaling(app_id, false);
      if (departure.first != nullptr)
      {
        on_stop_(app_id, *departure.first, *departure.second);
        return true;
      }
    }
    else if (ratio > threshold)
    {
      auto upscaling = Upscaling(app_id);
      if (upscaling.first != nullptr)
      {
        on_deploy_(app_id, *upscaling.first, *upscaling.second);
        locker_[app_id] = 5;
        return true;
      }
    }
    return false;
  }

  std::pair<Model *, Worker *> Upscaling(const string &app_id)
  {
    std::vector<Worker *> _workers = datastore_->get_workers();
    if (_workers.empty())
      return {nullptr, nullptr};

    std::vector<std::string> names;
    for (const auto name : datastore_->get_registered(app_id))
    {
      names.push_back(name);
    }
    return scheduler_->schedule(_workers, names);
  }

  std::pair<Model *, Worker *> Downscaling(const string &app_id, bool force)
  {
    std::vector<std::pair<Model *, Worker *>> candidates = datastore_->get_variant_workers(app_id);
    if (candidates.size() > 1)
    {
      if (force)
      {
        std::sort(candidates.begin(), candidates.end(),
                  [](std::pair<Model *, Worker *> &a, std::pair<Model *, Worker *> &b)
                  {
                    return a.first->get_throughput() < b.first->get_throughput();
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