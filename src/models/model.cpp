#include "model.hpp"

Model::Model() {}

Model::Model(const std::string &name, const int batch_size, const int device) : name_(name), batch_size_(batch_size)
{
  const std::string path = "/home/yusufaye/roomie-cpp/src/data/models/" + name + ".pt";
  module_ = torch::jit::load(path);

  torch::Device cuda_device = torch::Device(torch::kCUDA, device);
  // Move the module to the CUDA device
  module_.to(cuda_device);

  // Create a tensor
  input_ = torch::rand({batch_size, 3, 224, 224});

  // Move the tensor to the CUDA device
  input_ = input_.to(cuda_device);
}
at::Tensor Model::forward()
{
  // Run inference
  at::Tensor output = module_.forward({input_}).toTensor();
  return output;
}
at::Tensor Model::forward(cudaStream_t stream)
{
  // Run inference
  // at::cuda::CUDAStreamGuard guard(stream);
  at::Tensor output = module_.forward({input_}).toTensor();
  return output;
}