#ifndef MODEL_H
#define MODEL_H

#include <torch/script.h>
#include <cuda_runtime.h>

class Model
{
private:
  std::string name_;
  int batch_size_;
  torch::jit::script::Module module_;
  torch::Tensor input_;

public:
  Model();
  Model(const std::string &name, const int batch_size = 1, const int device = 0);
  at::Tensor forward();
  at::Tensor forward(cudaStream_t stream);
};

#endif  // MODEL_H