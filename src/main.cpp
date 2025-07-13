#include <iostream>
#include "manager/engine.h"
#include "manager/worker.h"
#include "manager/controller.h"
#include "manager/poisson_zipf_query_generator.h"
#include "utils/profiler.h"
#include "utils/datastore.h"
#include "networking/port.h"
#include "networking/message.h"

int main(int argc, char const *argv[])
{
  Engine *engine;
  if (argc > 1)
  {
    std::string engine_arg = argv[1];
    // std::string path_arg = argv[2];
    std::string path_arg;
    if (engine_arg == "Q" || engine_arg == "q")
    {
      path_arg = "/home/yusufaye/roomie-cpp/src/config/infaas/query_generator-howsrv-3.json";
      engine = new PoissonZipfQueryGenerator();
    }
    else if (engine_arg == "C" || engine_arg == "c")
    {
      path_arg = "/home/yusufaye/roomie-cpp/src/config/infaas/controller-howsrv-3.json";
      engine = new Controller(Approach::INFAAS);
    }
    else if (engine_arg == "W" || engine_arg == "w")
    {
      path_arg = "/home/yusufaye/roomie-cpp/src/config/infaas/worker_howsrv-3.json";
      engine = new WorkerEngine();
    }
    else
    {
      std::cerr << "⛔️[ERROR] Please provide a valide option, given is " << engine_arg << std::endl;
      exit(1);
    }
    engine->configure(path_arg);
    engine->start();
  }
  else
  {
    std::cerr << "⛔️[ERROR] Please provide a valide option" << std::endl;
    exit(1);
  }
  return 0;
}