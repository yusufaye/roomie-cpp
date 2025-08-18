#ifndef CONSTANT_H
#define CONSTANT_H

#include <vector>
#include <array>
#include <string>

const int MAX_GPU_MEMORY_OCCUPANCY = 50;
// const int MAX_GPU_MEMORY_OCCUPANCY = 90;

const std::vector<int> BATCH_SIZES = { 8, 16, 32 };

const std::string WORKDIR = "/usmb/roomie/";


#endif  // CONSTANT_H