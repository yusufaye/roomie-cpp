#ifndef OCCUPANCY_H
#define OCCUPANCY_H


#include <map>
#include <math.h>
#include "csv.h"

using namespace std;

class Perf
{
public:
  float occupancy = 0.0;
  int max_blocks = 0;
  int list_max_blocks[3] = {0, 0, 0};
  map<std::string, int> resource_required_per_block;
  int warpsPerBlock = 0;
  int warpsPerMultiprocessor = 0;
  Perf()
  {
    resource_required_per_block["warps_per_block"] = 0;
    resource_required_per_block["regs_per_block"] = 0;
    resource_required_per_block["shared_memory_per_block"] = 0;
  }

  std::array<float, 2> compute_perf_drop(int blocksPerSM)
  {
    // """Compute the performance drop.

    // Args:
    //   blocksPerSM (int): maximum blocks per multiprocessor.

    // Returns:
    //   tuple: (new theorical occupancy, ratio)
    // """
    int active_warps_per_SM = blocksPerSM * warpsPerBlock;
    float theoretical_occupancy = active_warps_per_SM / warpsPerMultiprocessor;
    return {theoretical_occupancy, theoretical_occupancy / occupancy};
  }
};

class NvidiaGpuSpec
{
public:
  int threadsPerWarp;
  int warpsPerMultiprocessor;
  int threadBlocksPerMultiprocessor;
  int sharedMemoryPerMultiprocessor;
  int registerFileSize;
  int registerAllocationUnitSize;
  int maxRegsPerThread;
  int maxRegsPerBlock;
  int sharedMemoryAllocationUnitSize;
  int warpAllocationGranularity;
  string limitedby[3] = {"Warp", "Register", "Shared Memory"};
  NvidiaGpuSpec(int major, int minor)
  {
    string computeCapability = major + "." + minor;
    string pathname = "data/gpu/gpu-configs.csv";
    // csv::CSVReader reader(pathname);
    // for (csv::CSVRow& row : reader)
    // {
    //   if (row["compute_capability"].get<string>() == computeCapability)
    //   {
    //     threadsPerWarp = row["threadsPerWarp"].get<int>();
    //     warpsPerMultiprocessor = row["warpsPerMultiprocessor"].get<int>();
    //     threadBlocksPerMultiprocessor = row["threadBlocksPerMultiprocessor"].get<int>();
    //     sharedMemoryPerMultiprocessor = row["sharedMemoryPerMultiprocessor"].get<int>();
    //     registerFileSize = row["registerFileSize"].get<int>();
    //     registerAllocationUnitSize = row["registerAllocationUnitSize"].get<int>();
    //     maxRegsPerThread = row["maxRegsPerThread"].get<int>();
    //     maxRegsPerBlock = row["maxRegsPerBlock"].get<int>();
    //     sharedMemoryAllocationUnitSize = row["sharedMemoryAllocationUnitSize"].get<int>();
    //     warpAllocationGranularity = row["warpAllocationGranularity"].get<int>();
    //     break;
    //   }
    // }
  }

  std::array<int, 3> boundaries()
  {
    // """Return the GPU boundaries such as the warps per multiprocessor, the register per block, and shared memory per block.

    // Returns:
    //   List[int]: ["warps_per_block", "regs_per_block", "shared_memory_per_block"]
    // """
    return {warpsPerMultiprocessor, registerFileSize, sharedMemoryPerMultiprocessor};
  }

  float Ceil(float a, float b)
  {
    return ceil(a / b) * b;
  }

  float Floor(float a, float b)
  {
    return floor(a / b) * b;
  }

  int Argmin(const int elements[], int size)
  {
    int a_min(0);
    int value = elements[0];
    for (size_t i = 1; i < size; i++)
      if (elements[i] < elements[a_min])
      {
        value = elements[i];
        a_min = i;
      }
    return a_min;
  }

  Perf theoretical_occupancy(
      int threadsPerBlock,
      int regsPerThread,
      int sharedMemory,
      bool verbose = false)
  {

    // """Compute gpu occupancy

    // Args:
    //   threadsPerBlock (int): Threads Per Block
    //   regsPerThread (int): Registers Per Thread
    //   sharedMemory (int): User Shared Memory Per Block
    //   verbose (bool, optional): _description_. Defaults to False.

    // Returns:
    //   _type_: _description_
    // """
    Perf perf;
    // compute the number of warps
    int warpsPerBlock = ceil(threadsPerBlock / threadsPerWarp);

    perf.resource_required_per_block["warps_per_block"] = warpsPerBlock;
    if (verbose)
      std::cout << "\tWarps per block: " << warpsPerBlock << std::endl;

    //
    // Limitation due to Warps
    int maxBlocksDueToWarps = fmin(
        threadBlocksPerMultiprocessor,
        floor(warpsPerMultiprocessor / warpsPerBlock) // # -> number of blocks with respect to the number of warps per block
    );

    // ##
    // # Limitation due to Registers
    if (verbose)
      std::cout << "Maximum block due to registers" << std::endl;

    int maxBlocksDueToRegs(0);
    if (regsPerThread > maxRegsPerThread)
    {
      if (verbose)
        std::cout << "\t\u274CError kernel launch" << std::endl;
      maxBlocksDueToRegs = 0;
    }
    else
    {
      // # the number of register per warp rounder up to the register allocation unit size
      int regsPerWarp = Ceil(regsPerThread * threadsPerWarp, registerAllocationUnitSize);
      if (verbose)
        std::cout << "\tRegister per warp: " << regsPerWarp << std::endl;

      // # register per block
      int regsPerBlock = regsPerWarp * warpsPerBlock;

      perf.resource_required_per_block["regs_per_block"] = regsPerBlock;
      if (verbose)
        std::cout << "\tRegister per block: " << regsPerBlock << std::endl;

      if (regsPerThread > 0)
      {
        // The number of maximum active warps per multiprocessor given the warp allocation granularity
        int warpsPerMultiprocessorLimitedByRegisters = Floor(
            maxRegsPerBlock / regsPerWarp, // # maximum warps per block with respect to the maxmimum register allocated per block
            warpAllocationGranularity);
        if (verbose)
          std::cout << "\tWarps per multiprocessor limited by registers: " << warpsPerMultiprocessorLimitedByRegisters << std::endl;
        // # The number of blocks limited by registers per warps times the factor of the maximum register that a block can use.
        // #  - for instance if a block can use at most half of the total register file, so we will have twice as much block,
        // #  - however, if a block can use up to the value of the register file, so the line is ignored.
        maxBlocksDueToRegs = floor(warpsPerMultiprocessorLimitedByRegisters / warpsPerBlock) * floor(registerFileSize / maxRegsPerBlock);
      }
      else
        maxBlocksDueToRegs = threadBlocksPerMultiprocessor;
    }

    if (verbose)
      std::cout << "\t\u274E Max blocks due to registers: " << maxBlocksDueToRegs << std::endl;

    // ##
    // # Limitation due to Shared Memory
    int maxBlocksDueToSMEM(threadBlocksPerMultiprocessor);
    if (sharedMemory > 0)
    {
      sharedMemory = Ceil(sharedMemory + 1024, sharedMemoryAllocationUnitSize);
      maxBlocksDueToSMEM = floor(sharedMemoryPerMultiprocessor / sharedMemory);
    }

    perf.resource_required_per_block["shared_memory_per_block"] = sharedMemory;

    int maxBlocks[3] = {maxBlocksDueToWarps, maxBlocksDueToRegs, maxBlocksDueToSMEM};
    int argmin = Argmin(maxBlocks, 3);
    if (verbose)
      std::cout << "Max Blocks Due To Warps: " << maxBlocksDueToWarps << "\nMax Blocks Due To Regs: " << maxBlocksDueToRegs << "\nMax Blocks Due To SMEM: " << maxBlocksDueToSMEM << std::endl;

    int blocksPerSM = maxBlocks[argmin];

    int active_warps_per_SM = blocksPerSM * warpsPerBlock;

    float theoretical_occupancy = active_warps_per_SM / warpsPerMultiprocessor;

    if (verbose)
      std::cout << "Limited by " << limitedby[argmin] << ", theoretical_occupancy: " << theoretical_occupancy << std::endl;

    perf.occupancy = theoretical_occupancy;
    perf.max_blocks = blocksPerSM;
    perf.list_max_blocks[0] = maxBlocks[0];
    perf.list_max_blocks[1] = maxBlocks[1];
    perf.list_max_blocks[2] = maxBlocks[2];
    perf.warpsPerBlock = warpsPerBlock;
    perf.warpsPerMultiprocessor = warpsPerMultiprocessor;
    return perf;
  }
};


#endif  // OCCUPANCY_H