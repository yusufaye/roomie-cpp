#ifndef BASE_SCHEDULER_H
#define BASE_SCHEDULER_H

#include <map>
#include "utils/datastore.h"
#include "utils/profiler.h"

class Scheduler
{
protected:
  std::map<std::string, Model *> cache;

public:
  Scheduler() {}
  virtual std::pair<Model *, Worker *> schedule(std::vector<Worker *> &workers, std::vector<std::string> &variant_candidates) = 0;

  Model *load_model_metadata(string hardware_platform, string variant_name)
  {
    std::string key = hardware_platform + "_" + variant_name;
    auto it = cache.find(key);
    if (it != cache.end())
    {
      return it->second;
    }

    Model *model = new Model(0, variant_name, hardware_platform);
    pre_profiled(*model);
    cache[key] = model;
    return model;
  }
};

#endif // BASE_SCHEDULER_H