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
    throw std::out_of_range("Vector is empty");
  }

  return std::distance(vec.begin(), std::min_element(vec.begin(), vec.end()));
}

bool bernoulli(float prob)
{
  // Create a random number generator
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> uniformDis(0.0, 1.0);
  double randomNumber = uniformDis(gen);
  if (randomNumber < prob)
  {
    return true;
  }
  return false;
}

float median(const std::vector<float> &vec)
{
  std::vector<float> sortedVec = vec;
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

double median(const std::vector<double> &vec)
{
  std::vector<double> sortedVec = vec;
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

float mean(const std::vector<float> &arr)
{
  float total = 0.0;
  for (const float item : arr)
  {
    total += item;
  }
  return total / arr.size();
}

float mean(const std::vector<double> &arr)
{
  double total = 0.0;
  for (const double item : arr)
  {
    total += item;
  }
  return total / arr.size();
}

std::string vec2str(const std::vector<double> &arr)
{
  std::string out = "[ ";
  for (size_t i = 0; i < arr.size(); i++)
  {
    if (i > 0)
      out += ", ";
    out += std::to_string(arr[i]);
  }
  out += " ]";
  return out;
}

std::string vec2str(const std::vector<float> &arr)
{
  std::string out = "[ ";
  for (size_t i = 0; i < arr.size(); i++)
  {
    if (i > 0)
      out += ", ";
    out += std::to_string(arr[i]);
  }
  out += " ]";
  return out;
}

std::string vec2str(const std::vector<std::string> &arr)
{
  std::string out = "[ ";
  for (size_t i = 0; i < arr.size(); i++)
  {
    if (i > 0)
      out += ", ";
    out += arr[i];
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