#include <unistd.h>
#include <iostream>
#include <nlohmann/json.hpp>
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
  Engine *engine;
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
      std::cerr << "⛔️ Error loading configuration" << "\n\t" << e.what() << '\n';
      return 1;
    }
    std::string type = config["type"];

    std::cout << "⚠️[" + type + "] About to load configuration from " + "'" + path + "'" << std::endl;
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
      std::cerr << "⛔️[ERROR] Please provide a valide engine, given is " << type << std::endl;
      exit(1);
    }
    engine->configure(config);
    engine->start();
  }
  else
  {
    std::cerr << "⛔️[ERROR] Please provide a valide option" << std::endl;
    exit(1);
  }
  return 0;
}