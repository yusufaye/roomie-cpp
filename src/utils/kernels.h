#ifndef KERNELS_H
#define KERNELS_H

#include <iostream>
#include <map>
#include <string>
#include "occupancy.h"

class NcuKernel
{
public:
  NcuKernel() {}
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

  float thread_block()
  {
    return block_dim_x * block_dim_y * block_dim_z;
  }
};

class Operation : public NcuKernel
{
private:
  Perf perf_;

public:
  int xxx_order;
  int xxx_max_blocks_granted;
  int xxx_duration;
  int xxx_extended_duration;
  int xxx_additional_duration;

  Operation()
  {
    reset();
  }

  Perf get_perf()
  {
    return perf_;
  }

  void set_perf(Perf perf)
  {
    perf_ = perf;
  }

  float new_occupancy()
  {
    return (xxx_max_blocks_granted * perf_.warpsPerBlock) / perf_.warpsPerMultiprocessor * 100;
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

  map<std::string, int> resource_required_per_block()
  {
    return perf_.resource_required_per_block;
  }

  std::array<int, 4> resources_per_block()
  {
    // """Return the GPU resources required such as the warps per multiprocessor, the register per block, and shared memory per block.

    // Returns:
    //   List[int]: ["warps_per_block", "regs_per_block", "shared_memory_per_block"]
    // """
    return {perf_.resource_required_per_block["warps_per_block"], perf_.resource_required_per_block["regs_per_block"], perf_.resource_required_per_block["shared_memory_per_block"]};
  }

  float duration_after_interference()
  {
    // """Duration with respect to the interference.

    // Returns:
    //     float: _description_
    // """
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
};

#endif  // KERNELS_H