#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <sstream>
#include <chrono>
#include <iomanip>

enum class LogLevel
{
  DEBUG,
  INFO,
  WARNING,
  ERROR
};

class Logging
{
public:
  Logging(LogLevel level = LogLevel::INFO) : currentLevel(level) {}

  void setLogLevel(LogLevel level)
  {
    currentLevel = level;
  }

  template <LogLevel level>
  void log(const std::string &message)
  {
    if (level >= currentLevel)
    {
      logImpl(level, message);
    }
  }

  template <LogLevel level, typename... Args>
  void log(const std::tuple<Args...> &data)
  {
    if (level >= currentLevel)
    {
      logImpl(level, data);
    }
  }

private:
  LogLevel currentLevel;

  void logImpl(LogLevel level, const std::string &message)
  {
    std::string levelStr;
    switch (level)
    {
    case LogLevel::DEBUG:
      levelStr = "DEBUG";
      break;
    case LogLevel::INFO:
      levelStr = "INFO";
      break;
    case LogLevel::WARNING:
      levelStr = "WARNING";
      break;
    case LogLevel::ERROR:
      levelStr = "ERROR";
      break;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << " [" << levelStr << "] " << message << std::endl;
  }

  template <typename... Args>
  void logImpl(LogLevel level, const std::tuple<Args...> &data)
  {
    std::string levelStr;
    switch (level)
    {
    case LogLevel::DEBUG:
      levelStr = "DEBUG";
      break;
    case LogLevel::INFO:
      levelStr = "INFO";
      break;
    case LogLevel::WARNING:
      levelStr = "WARNING";
      break;
    case LogLevel::ERROR:
      levelStr = "ERROR";
      break;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << " [" << levelStr << "] ";
    bool first = true;
    std::apply([&](auto &&...args)
               { (([&](auto &&arg)
                   {
                if (!first) {
                    std::cout << ", ";
                }
                std::cout << typeid(arg).name() << ": " << arg;
                first = false; }(args)),
                  ...); }, data);
    std::cout << std::endl;
  }
};

#endif // LOGGING_H