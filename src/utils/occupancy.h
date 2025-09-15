#ifndef OCCUPANCY_H
#define OCCUPANCY_H

#include <map>
#include <math.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "csv.h"

using namespace std;
using json = nlohmann::json;

class Perf
{
public:
  float occupancy = 0.0;
  int max_blocks = 0;
  std::array<int, 3> list_max_blocks = {0, 0, 0};
  std::map<std::string, int> resource_required_per_block;
  int warpsPerBlock = 0;
  int warpsPerMultiprocessor = 0;
  Perf()
  {
    resource_required_per_block["warps_per_block"] = 0;
    resource_required_per_block["regs_per_block"] = 0;
    resource_required_per_block["shared_memory_per_block"] = 0;
  }

  Perf(const Perf &perf)
  {
    occupancy = perf.occupancy;
    max_blocks = perf.max_blocks;
    list_max_blocks = perf.list_max_blocks;
    resource_required_per_block = perf.resource_required_per_block;
    warpsPerBlock = perf.warpsPerBlock;
    warpsPerMultiprocessor = perf.warpsPerMultiprocessor;
  }

  /**
   * Compute the performance drop.
   * Args:
   *  blocksPerSM (int): maximum blocks per multiprocessor.
   * Returns:
   *  pair: (new theorical occupancy, ratio)
   */
  std::pair<float, float> compute_perf_drop(int blocksPerSM)
  {
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
    string pathname = "data/gpu/gpu-configs.json";
    json j;
    try
    {
      // read a JSON file
      std::ifstream i(pathname);
      i >> j;
      i.close();
    }
    catch (const std::exception &e)
    {
      spdlog::error("⛔️ Error loading gpu configs from {}\n\t{}", pathname, e.what());
      return;
    }
    std::string computeCapability = std::to_string(major) + "." + std::to_string(minor);
    try
    {
      for (const auto &item : j)
      {
        if (item["computeCapability"] == computeCapability)
        {
          threadsPerWarp = item["threadsPerWarp"];
          warpsPerMultiprocessor = item["warpsPerMultiprocessor"];
          threadBlocksPerMultiprocessor = item["threadBlocksPerMultiprocessor"];
          sharedMemoryPerMultiprocessor = item["sharedMemoryPerMultiprocessor"];
          registerFileSize = item["registerFileSize"];
          registerAllocationUnitSize = item["registerAllocationUnitSize"];
          maxRegsPerThread = item["maxRegsPerThread"];
          maxRegsPerBlock = item["maxRegsPerBlock"];
          sharedMemoryAllocationUnitSize = item["sharedMemoryAllocationUnitSize"];
          warpAllocationGranularity = item["warpAllocationGranularity"];
          break;
        }
      }
    }
    catch(const std::exception& e)
    {
      spdlog::error("Error while loading GPU spec\n\t{}", e.what());
    }
  }

  /**
   * Return the GPU boundaries such as the warps per multiprocessor, the register per block, and shared memory per block.
   * Returns:
   *  List[int]: ["warps_per_block", "regs_per_block", "shared_memory_per_block"]
   */
  std::vector<int> boundaries() const
  {
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

  /**
   * Compute gpu occupancy.
   * Args:
   *  threadsPerBlock (int): Threads Per Block
   *  regsPerThread (int): Registers Per Thread
   *  sharedMemory (int): User Shared Memory Per Block
   *  verbose (bool, optional): _description_. Defaults to False.
   */
  Perf theoretical_occupancy(
      int threadsPerBlock,
      int regsPerThread,
      int sharedMemory,
      bool verbose = true)
  {
    Perf perf;
    // compute the number of warps
    int warpsPerBlock = ceil((float)threadsPerBlock / threadsPerWarp);
    spdlog::debug("\tthreadsPerWarp: {}", threadsPerWarp);
    perf.resource_required_per_block["warps_per_block"] = warpsPerBlock;
    if (verbose)
    {
      spdlog::debug("\tWarps per block: {}", warpsPerBlock);
    }

    //
    // Limitation due to Warps
    int maxBlocksDueToWarps = min(
        threadBlocksPerMultiprocessor,
        warpsPerMultiprocessor / warpsPerBlock // # -> number of blocks with respect to the number of warps per block
    );

    // ##
    // # Limitation due to Registers
    if (verbose)
    {
      spdlog::debug("Maximum block due to registers");
    }

    int maxBlocksDueToRegs(0);
    if (regsPerThread > maxRegsPerThread)
    {
      if (verbose)
      {
        spdlog::debug("\t\u274CError kernel launch");
      }
      maxBlocksDueToRegs = 0;
    }
    else
    {
      // # the number of register per warp rounder up to the register allocation unit size
      int regsPerWarp = Ceil(regsPerThread * threadsPerWarp, registerAllocationUnitSize);
      if (verbose)
      {
        spdlog::debug("\tRegister per warp: {}", regsPerWarp);
      }

      // # register per block
      int regsPerBlock = regsPerWarp * warpsPerBlock;

      perf.resource_required_per_block["regs_per_block"] = regsPerBlock;
      if (verbose)
      {
        spdlog::debug("\tRegister per block: {}", regsPerBlock);
      }

      if (regsPerThread > 0)
      {
        // The number of maximum active warps per multiprocessor given the warp allocation granularity
        int warpsPerMultiprocessorLimitedByRegisters = Floor(
            maxRegsPerBlock / regsPerWarp, // # maximum warps per block with respect to the maxmimum register allocated per block
            warpAllocationGranularity);
        if (verbose)
        {
          spdlog::debug("\tWarps per multiprocessor limited by registers: {}", warpsPerMultiprocessorLimitedByRegisters);
        }
        // # The number of blocks limited by registers per warps times the factor of the maximum register that a block can use.
        // #  - for instance if a block can use at most half of the total register file, so we will have twice as much block,
        // #  - however, if a block can use up to the value of the register file, so the line is ignored.
        maxBlocksDueToRegs = floor(warpsPerMultiprocessorLimitedByRegisters / warpsPerBlock) * floor(registerFileSize / maxRegsPerBlock);
      }
      else
        maxBlocksDueToRegs = threadBlocksPerMultiprocessor;
    }

    if (verbose)
    {
      spdlog::debug("\t\u274E Max blocks due to registers: {}", maxBlocksDueToRegs);
    }

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
      spdlog::debug("Max Blocks Due To Warps: {}\nMax Blocks Due To Regs: {}\nMax Blocks Due To SMEM: {}", maxBlocksDueToWarps, maxBlocksDueToRegs, maxBlocksDueToSMEM);

    int blocksPerSM = maxBlocks[argmin];

    int active_warps_per_SM = blocksPerSM * warpsPerBlock;

    float theoretical_occupancy = active_warps_per_SM / warpsPerMultiprocessor;

    if (verbose)
    {
      spdlog::debug("Limited by {}, theoretical_occupancy: {}", limitedby[argmin], theoretical_occupancy);
    }

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

#endif // OCCUPANCY_H