#ifndef LOAD_BALANCING_H
#define LOAD_BALANCING_H

#include <string>
#include <vector>
// #include <numeric>
#include <optional>
// #include <algorithm>
#include <unordered_map>

class WeightedRoundRobinScheduling
{
private:
  std::vector<std::string> keys_;
  std::unordered_map<std::string, int> weight_;
  int n = 0;
  int i = -1;
  int cw = 0;

public:
  void set(const std::string &key, int w = 1)
  {
    auto it = std::find(keys_.begin(), keys_.end(), key);
    if (it != keys_.end())
    {
      update(key, w);
      return;
    }
    keys_.push_back(key);
    weight_[key] = w;
    n = keys_.size();
  }

  void remove(const std::string &key)
  {
    keys_.erase(std::remove(keys_.begin(), keys_.end(), key), keys_.end());
    weight_.erase(key);
    n = keys_.size();
  }

  void update(const std::string &key, int w)
  {
    weight_[key] = w;
  }

  std::optional<std::string> next()
  {
    if (keys_.size() == 1)
      return keys_[0];

    while (true)
    {
      i = (i + 1) % n;
      if (i == 0)
      {
        // Compute GCD of all weight_s
        int g = weight_[keys_[0]];
        for (size_t k = 1; k < keys_.size(); ++k)
        {
          g = std::gcd(g, weight_[keys_[k]]);
        }
        cw -= g;

        if (cw <= 0)
        {
          auto maxIt = std::max_element(keys_.begin(), keys_.end(),
                                        [&](const std::string &a, const std::string &b)
                                        {
                                          return weight_[a] < weight_[b];
                                        });
          cw = weight_[*maxIt];
          if (cw == 0)
            return std::nullopt;
        }
      }

      if (weight_[keys_[i]] >= cw)
      {
        return keys_[i];
      }
    }
  }

  std::string to_string() const
  {
    std::string data_format = "loadb -> [";
    for (const auto key : keys_)
    {
      if (weight_.find(key) != weight_.end())
      {
        data_format += "'" + key + "': " + std::to_string(weight_.at(key)) + ", ";
      }
      else
      {
        data_format += "'" + key + "': null, ";
      }
    }
    data_format += "]";
    return data_format;
  }
};

class LoadBalancer
{
private:
  std::unordered_map<std::string, WeightedRoundRobinScheduling> loadBalancer;

public:
  void set(const std::string &app_id, const std::string &key, int w)
  {
    loadBalancer[app_id].set(key, w);
  }

  void remove(const std::string &app_id, const std::string &key)
  {
    if (loadBalancer.count(app_id))
    {
      loadBalancer[app_id].remove(key);
    }
  }

  void update(const std::string &app_id, const std::string &key, int w)
  {
    if (loadBalancer.count(app_id))
    {
      loadBalancer[app_id].update(key, w);
    }
  }

  std::optional<std::string> next(const std::string &app_id)
  {
    if (loadBalancer.count(app_id))
    {
      return loadBalancer[app_id].next();
    }
    return std::nullopt;
  }

  std::string to_string() const
  {
    std::string data_format = "LoadBalancing(";
    for (auto &[app_id, loadb] : loadBalancer)
    {
      data_format += "\n\t'" + app_id + "': " + loadb.to_string();
    }
    data_format += ")";
    return data_format;
  }
};

#endif // LOAD_BALANCING_H