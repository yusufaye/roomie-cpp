#ifndef CONSTANT_H
#define CONSTANT_H

#include <vector>
#include <array>
#include <string>

const int MAX_GPU_MEMORY_OCCUPANCY = 90;

const std::vector<int> BATCH_SIZES = { 32, 64, 128 };

const std::string WORKDIR = "/usmb/roomie/";


#endif  // CONSTANT_H