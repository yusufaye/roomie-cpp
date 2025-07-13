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

// TESTING NETWORK
// int main(int argc, char const *argv[])
// {
//   auto callback = [](const Message &message)
//   {
//     std::cout << "[Recv] " << message.to_string() << std::endl;
//   };

//   string host = "localhost";
//   int port = 8888;

//   OutPort out_port(0, host, port);
//   InPort in_port(host, port, callback);

//   std::map<std::string, std::string> data;
//   data["firstname"] = "yusuf";
//   data["lastname"]  = "faye";
//   for (size_t i = 0; i < 10; i++)
//   {
//     Message message("DEPLOY", data);
//     out_port.push(message);
//   }
//   data.clear();
//   out_port.push(Message("FINISHED", data));
// }

// void test_model()
// {
//   Model model(0, "alexnet", "xavier");
//   pre_profiled(model);
//   model.batch_size = 64;
//   for (auto &it : *model.get_Kernel())
//   {
//     cout << "Total kernels: " << it.first << " : " << it.second.size() << endl;
//   }
// }

// void run_server()
// {
//   WebSocketServer server("");
//   server.run();
// }

// int networkinging(int argc, char const *argv[])
// {
//   std::thread server_thread(run_server);

//   server_thread.detach();

//   WebSocketClient client("ws://localhost:9002");
//   std::thread client_thread([&client]
//                             { client.run(); });
//   // Wait for the connection to be established
//   while (!client.is_connected())
//   {
//     std::this_thread::sleep_for(std::chrono::milliseconds(100));
//   }

//   Message message("hello", "world");
//   for (int i; i < 10; i++)
//     client.send(message);
//   client_thread.join();

//   return 0;
// }
