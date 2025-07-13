#ifndef QUEUE_H
#define QUEUE_H


#include <queue>
#include <mutex>
#include <condition_variable>

#include <atomic>
#include <vector>
#include <iostream>

template <typename T>
class BlockingQueue
{
public:
  // Push an item into the queue
  void push(const T &item)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(item);
    }
    cond_var_.notify_one(); // Wake up one waiting thread
  }

  // Pop an item from the queue (blocks if empty)
  T pop()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [this]()
                   { return !queue_.empty(); });

    T item = queue_.front();
    queue_.pop();
    return item;
  }

  // Optional: Check size (non-blocking)
  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;
};

template <typename T>
class HybridSPSCQueue
{
public:
  HybridSPSCQueue(size_t capacity)
      : buffer_(capacity), capacity_(capacity), head_(0), tail_(0) {}

  void push(const T &item)
  {
    size_t next_tail = (tail_ + 1) % capacity_;
    if (next_tail == head_.load(std::memory_order_acquire))
    {
      throw std::runtime_error("Queue overflow"); // Or resize if needed
    }

    buffer_[tail_] = item;
    tail_ = next_tail;

    // Notify consumer if it was waiting
    std::lock_guard<std::mutex> lock(mutex_);
    cond_var_.notify_one();
  }

  T pop_blocking()
  {
    while (true)
    {
      if (head_.load(std::memory_order_acquire) != tail_)
      {
        T item = buffer_[head_];
        head_ = (head_ + 1) % capacity_;
        return item;
      }

      // Wait until producer pushes something
      std::unique_lock<std::mutex> lock(mutex_);
      cond_var_.wait(lock, [this]()
                     { return head_.load(std::memory_order_acquire) != tail_; });
    }
  }

private:
  std::vector<T> buffer_;
  const size_t capacity_;
  std::atomic<size_t> head_;
  size_t tail_; // Only producer writes to tail

  std::mutex mutex_;
  std::condition_variable cond_var_;
};


#endif // QUEUE_H