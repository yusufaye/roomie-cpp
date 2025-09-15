#ifndef GENERAL_H
#define GENERAL_H

#include <mutex>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <condition_variable>

template <typename T>
std::vector<size_t> argsort(const std::vector<T> &vec)
{
  std::vector<std::pair<T, size_t>> paired_vec;
  paired_vec.reserve(vec.size());
  for (size_t i = 0; i < vec.size(); ++i)
  {
    paired_vec.emplace_back(vec[i], i);
  }

  std::sort(paired_vec.begin(), paired_vec.end(),
            [](const auto &a, const auto &b)
            { return a.first < b.first; });

  std::vector<size_t> sorted_indices;
  sorted_indices.reserve(vec.size());
  for (const auto &pair : paired_vec)
  {
    sorted_indices.push_back(pair.second);
  }

  return sorted_indices;
}

template <typename T>
void reorder_vector(std::vector<T> &vec, const std::vector<size_t> &indices)
{
  std::vector<T> temp(vec.size());
  for (size_t i = 0; i < vec.size(); ++i)
  {
    temp[i] = vec[indices[i]];
  }
  vec = temp;
}

template <typename T>
size_t argmin(const std::vector<T> &vec)
{
  if (vec.empty())
  {
    throw std::out_of_range("[Argmin] Vector is empty");
  }

  return std::distance(vec.begin(), std::min_element(vec.begin(), vec.end()));
}

template <typename T>
size_t argmax(const std::vector<T> &vec)
{
  if (vec.empty())
  {
    throw std::out_of_range("[Argmax] Vector is empty");
  }

  return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

bool bernoulli(float prob)
{
  // static std::random_device rd;
  // static std::mt19937 gen(rd());
  static std::mt19937 gen(987654);
  std::uniform_real_distribution<double> uniformDis(0.0, 1.0);
  double randomNumber = uniformDis(gen);
  if (randomNumber < prob)
  {
    return true;
  }
  return false;
}

std::vector<std::vector<double>> create_mask(const std::vector<double> &vec)
{
  int L = vec.size();
  int M = std::min((int)(L / 2.0), 20);
  M = (M % 2 == 0) ? M + 1 : M;
  std::vector<std::vector<double>> mask(M, std::vector<double>(L, 1.0));
  int max_pad = M / 2;
  for (int pad = 1; pad <= max_pad; ++pad)
  {
    for (int j = 0; j < pad; ++j)
    {
      mask[pad - 1][j] = 0.0;
    }
    for (int j = L - pad; j < L; ++j)
    {
      mask[M - pad][j] = 0.0;
    }
  }
  std::vector<std::vector<double>> result(M, std::vector<double>(L, 0.0));
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < L; ++j)
    {
      result[i][j] = vec[j] * mask[i][j];
    }
  }

  // e.g., of mask an array {1, 2, 3, 4, 5, 6, 7, 8}
  // [0 2 3 4 5 6 7 8 ]
  // |0 0 3 4 5 6 7 8 |
  // |1 2 3 4 5 6 7 8 |
  // |1 2 3 4 5 6 0 0 |
  // [1 2 3 4 5 6 7 0 ]
  return result;
}

std::vector<std::vector<bool>> create_bool_mask(int L, int M)
{
  M = std::min(M, 20);
  M = (M % 2 == 0) ? M + 1 : M;
  std::vector<std::vector<bool>> mask(M, std::vector<bool>(L, true));
  int max_pad = M / 2;
  for (int pad = 1; pad <= max_pad; ++pad)
  {
    for (int j = 0; j < pad; ++j)
    {
      mask[pad - 1][j] = false;
    }
    for (int j = L - pad; j < L; ++j)
    {
      mask[M - pad][j] = false;
    }
  }
  return mask;
}

template <typename T>
T median(const std::vector<T> &vec)
{
  if (vec.empty())
  {
    throw std::out_of_range("[Median] Vector is empty");
  }
  std::vector<T> sortedVec = vec;
  std::sort(sortedVec.begin(), sortedVec.end());
  size_t n = sortedVec.size();
  if (n % 2 == 0)
  {
    return (sortedVec[n / 2 - 1] + sortedVec[n / 2]) / 2;
  }
  else
  {
    return sortedVec[n / 2];
  }
}

template <typename T>
T mean(const std::vector<T> &vec)
{
  if (vec.empty())
  {
    throw std::out_of_range("[Mean] Vector is empty");
  }
  T total = 0.0;
  for (const T item : vec)
  {
    total += item;
  }
  return total / vec.size();
}

template <typename T>
T minimum(const std::vector<T> &vec)
{
  if (vec.empty())
  {
    throw std::out_of_range("[Minimum] Vector is empty");
  }
  T min_v = vec[0];
  for (size_t i = 1; i < vec.size(); i++)
  {
    if (vec[i] < min_v)
    {
      min_v = vec[i];
    }
  }

  return min_v;
}

template <typename T>
T maximum(const std::vector<T> &vec)
{
  if (vec.empty())
  {
    throw std::out_of_range("[Maximum] Vector is empty");
  }
  T max_v = vec[0];
  for (size_t i = 1; i < vec.size(); i++)
  {
    if (vec[i] > max_v)
    {
      max_v = vec[i];
    }
  }

  return max_v;
}

template <typename T>
std::string vec2str(const std::vector<T> &vec)
{
  if (vec.empty())
  {
    return "[]";
  }
  std::string out = "[ ";
  for (size_t i = 0; i < vec.size(); i++)
  {
    if (i > 0)
      out += ", ";
    out += std::to_string(vec[i]);
  }
  out += " ]";
  return out;
}

std::string vec2str(const std::vector<std::string> &vec)
{
  if (vec.empty())
  {
    return "[]";
  }
  std::string out = "[ ";
  for (size_t i = 0; i < vec.size(); i++)
  {
    if (i > 0)
      out += ", ";
    out += vec[i];
  }
  out += " ]";
  return out;
}

class RandomGenerator
{
public:
  RandomGenerator(int seed = 1234)
      : engine_(seed), distribution_(999, 9999) {}

  int next()
  {
    while (true)
    {
      int value = distribution_(engine_);
      if (used_values_.insert(value).second)
      {
        return value;
      }
    }
  }

  void remove(int value)
  {
    used_values_.erase(value);
  }

private:
  std::mt19937 engine_;
  std::uniform_int_distribution<int> distribution_;
  std::unordered_set<int> used_values_;
};

class Event
{
public:
  Event() : is_set_(false) {}

  void set()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_set_ = true;
    cv_.notify_all();
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_set_ = false;
  }

  void wait()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]()
             { return is_set_; });
  }

  bool is_set() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_set_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool is_set_;
};

#endif // GENERAL_H