#include "network/message.h"
#include "network/websocket.h"
#include "utils/profiler.h"

int main(int argc, char const *argv[])
{
  vector<NcuKernel> kernels = get_profiled_data("alexnet");
  cout << "Total kernels: " << kernels.size() << endl;
  return 0;
}

void run_server()
{
  WebSocketServer server("");
  server.run();
}

int networking(int argc, char const *argv[])
{
  std::thread server_thread(run_server);

  server_thread.detach();

  WebSocketClient client("ws://localhost:9002");
  std::thread client_thread([&client]
                            { client.run(); });
  // Wait for the connection to be established
  while (!client.is_connected())
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  Message message("hello", "world");
  for (int i; i < 10; i++)
    client.send(message);
  client_thread.join();

  return 0;
}
