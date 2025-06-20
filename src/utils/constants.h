#ifndef CONSTANT_H
#define CONSTANT_H

#include <array>
#include <string>

const int MAX_GPU_MEMORY_OCCUPANCY = 90;

// NVIDIA A100
const int BATCH_SIZES[3] = { 32, 64, 128 };

const std::string WORKDIR = "/home/yusufaye/roomie-cpp/src/";


#endif  // CONSTANT_H