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

int main(int argc, char const *argv[])
{
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_color_mode(spdlog::color_mode::always);
  auto logger = std::make_shared<spdlog::logger>("roomie", console_sink);
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::debug);

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
      spdlog::error("{} {} {}", "⛔️ Error loading configuration", "\n\t", e.what());
      return 1;
    }
    std::string type = config["type"];

    spdlog::info("[{}] About to load configuration from '{}'", type, path);
    // std::cout << "⚠️[" + type + "] About to load configuration from " + "'" + path + "'" << std::endl;
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
    engine->start();
  }
  else
  {
    spdlog::error("⛔️[ERROR] Please provide a valide option");
    exit(1);
  }
  return 0;
}