#ifndef GENERAL_H
#define GENERAL_H


#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <condition_variable>

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



class Event {
public:
    Event() : is_set_(false) {}

    void set() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_set_ = true;
        cv_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_set_ = false;
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return is_set_; });
    }

    bool is_set() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_set_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool is_set_;
};


#endif // GENERAL_H