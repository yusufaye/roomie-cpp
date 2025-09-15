#ifndef KERNELS_H
#define KERNELS_H

#include <map>
#include <string>
#include <iostream>
#include "occupancy.h"

class Kernel
{
public:
  std::string kernel_name;
  // Grid dims
  int grid_dim_x;
  int grid_dim_y;
  int grid_dim_z;
  // Block dims
  int block_dim_x;
  int block_dim_y;
  int block_dim_z;
  int register_per_thread;
  float duration;
  // Static shared memory size per block, allocated for the kernel.
  float static_shared_memory_per_block;
  // Dynamic shared memory size per block, allocated for the kernel.
  float dynamic_shared_memory_per_block;
  float threads;
  // Number of waves per SM. Partial waves can lead to tail effects where some SMs become idle while others still have pending work to complete.
  float waves_per_sm;
  // Shared memory size configured for the kernel launch. The size depends on the static, dynamic, and driver shared memory requirements as well as the specified or platform-determined configuration size.
  float shared_memory;
  float theoretical_occupancy;
  float theoretical_active_warps_per_SM;
  float achieved_occupancy;
  float achieved_active_warps_per_SM;
  float block_limit_registers;
  float block_limit_shared_mem;
  float block_limit_warps;
  float block_limit_sm;
  float capability_minor;
  float capability_major;

  Kernel() {}

  Kernel(const Kernel &kernel)
  {
    kernel_name = kernel.kernel_name;
    grid_dim_x = kernel.grid_dim_x;
    grid_dim_y = kernel.grid_dim_y;
    grid_dim_z = kernel.grid_dim_z;
    block_dim_x = kernel.block_dim_x;
    block_dim_y = kernel.block_dim_y;
    block_dim_z = kernel.block_dim_z;
    register_per_thread = kernel.register_per_thread;
    duration = kernel.duration;
    static_shared_memory_per_block = kernel.static_shared_memory_per_block;
    dynamic_shared_memory_per_block = kernel.dynamic_shared_memory_per_block;
    threads = kernel.threads;
    waves_per_sm = kernel.waves_per_sm;
    shared_memory = kernel.shared_memory;
    theoretical_occupancy = kernel.theoretical_occupancy;
    theoretical_active_warps_per_SM = kernel.theoretical_active_warps_per_SM;
    achieved_occupancy = kernel.achieved_occupancy;
    achieved_active_warps_per_SM = kernel.achieved_active_warps_per_SM;
    block_limit_registers = kernel.block_limit_registers;
    block_limit_shared_mem = kernel.block_limit_shared_mem;
    block_limit_warps = kernel.block_limit_warps;
    block_limit_sm = kernel.block_limit_sm;
    capability_minor = kernel.capability_minor;
    capability_major = kernel.capability_major;
    perf_ = kernel.perf_;
  }

  float thread_block()
  {
    return block_dim_x * block_dim_y * block_dim_z;
  }

  /* SPECIFIC TO ROOMIE */

  int xxx_order = 0;
  int xxx_max_blocks_granted = 0;
  float xxx_duration = 0.0;
  float xxx_extended_duration = 0.0;
  float xxx_additional_duration = 0.0;

  Perf get_perf()
  {
    return perf_;
  }

  void set_perf(Perf perf)
  {
    perf_ = perf;
  }

  float new_occupancy() const
  {
    return ((float)xxx_max_blocks_granted * perf_.warpsPerBlock) / perf_.warpsPerMultiprocessor * 100;
  }

  int order()
  {
    return xxx_order;
  }

  void set_order(int order)
  {
    xxx_order = order;
  }

  int max_blocks()
  {
    return perf_.max_blocks;
  }

  // std::map<std::string, int> resource_required_per_block()
  // {
  //   return perf_.resource_required_per_block;
  // }

  /**
   * Return the GPU resources required such as the warps per multiprocessor, the register per block, and shared memory per block.
   * Returns:
   *  List[int]: ["warps_per_block", "regs_per_block", "shared_memory_per_block"]
   */
  std::vector<int> resource_required_per_block()
  {
    return {
        perf_.resource_required_per_block["warps_per_block"],
        perf_.resource_required_per_block["regs_per_block"],
        perf_.resource_required_per_block["shared_memory_per_block"],
    };
  }

  /**
   * Duration with respect to the interference.
   */
  float duration_after_interference()
  {
    if (xxx_additional_duration < 0)
    {
      spdlog::error("[Kernel] duration must be positive but {}", xxx_additional_duration);
      throw invalid_argument("Kernel interference error.");
    }
    return duration + xxx_additional_duration;
  }

  void reset()
  {
    xxx_order = 0;
    xxx_max_blocks_granted = 0;
    xxx_duration = duration;
    xxx_extended_duration = duration; // additional duration
    xxx_additional_duration = 0.0;    // additional duration
  }

private:
  Perf perf_;
};

#endif // KERNELS_H