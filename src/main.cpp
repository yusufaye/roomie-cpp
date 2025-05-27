#include "models/model.hpp"

void infer(Model& model) {
  while (true)
  {
    at::Tensor output = model.forward();
  }
  
  // std::cout << output << std::endl;
}

int main(int argc, char const *argv[])
{
    // Create input tensor
  torch::Tensor input_tensor = torch::rand({1, 5});

  // Create multiple models
  Model models[5];
  for (int i = 0; i < 5; i++) {
    models[i] = Model("resnet");
  }

  // Create threads to perform inference in parallel
  std::thread threads[5];
  for (int i = 0; i < 5; i++) {
    threads[i] = std::thread(infer, std::ref(models[i]));
  }

  // Wait for all threads to finish
  for (int i = 0; i < 5; i++) {
    threads[i].join();
  }

  return 0;
}
