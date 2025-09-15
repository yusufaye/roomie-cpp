#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "manager/engine.h"
#ifdef CUDA_AVAILABLE
#include "manager/worker.h"
#endif
#include "manager/controller.h"
#include "manager/poisson_zipf_query_generator.h"
#include "utils/profiler.h"
#include "utils/datastore.h"
#include "networking/port.h"
#include "networking/message.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

std::atomic<bool> terminateFlag(false);

void signalHandler(int sig)
{
  spdlog::debug("Signal handler received: {}", sig);
  terminateFlag.store(true);
}

int main(int argc, char const *argv[])
{
  std::signal(SIGABRT, signalHandler);
  std::signal(SIGSEGV, signalHandler);
  std::signal(SIGFPE, signalHandler);
  std::signal(SIGILL, signalHandler);
  std::signal(SIGINT, signalHandler);  // Ctrl+C
  std::signal(SIGTERM, signalHandler); // kill

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_color_mode(spdlog::color_mode::always);
  auto logger = std::make_shared<spdlog::logger>("roomie", console_sink);
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::info);

  if (argc > 1)
  {
    std::string path = argv[1];
    json config;
    try
    {
      std::ifstream i(path);
      i >> config;
      i.close();
    }
    catch (const std::exception &e)
    {
      spdlog::error("⛔️ Error loading configuration from {}\n\t{}", path, e.what());
      return 1;
    }
    std::string type = config["type"];

    spdlog::debug("[{}] About to load configuration from '{}'", type, path);
    Engine *engine;
    if (type == "PoissonZipfQueryGenerator")
    {
      engine = new PoissonZipfQueryGenerator();
    }
    else if (type == "Controller")
    {
      engine = new Controller();
    }
#ifdef CUDA_AVAILABLE
    else if (type == "WorkerEngine")
    {
      engine = new WorkerEngine();
    }
#endif
    else
    {
      spdlog::error("⛔️[ERROR] Please provide a valide engine, given is {}", type);
      exit(1);
    }
    engine->configure(config);
    std::thread start_thread([&engine]()
                             {
                              engine->start();
                            terminateFlag.store(true);
                           });

    while (!terminateFlag.load())
    {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    spdlog::debug("👋Will shutdown");
    engine->shutdown();
    exit(0);
  }
  else
  {
    spdlog::error("⛔️[ERROR] Please provide a valide option");
    exit(1);
  }
  return 0;
}