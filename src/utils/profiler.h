#ifndef PROFILER_H
#define PROFILER_H

// #include <format>
#include <string>
#include <iostream>
#include "csv.h"
#include "math.h"
#include "general.h"
#include "kernels.h"
#include "datastore.h"
#include "constants.h"


double convert(const std::string &v)
{
    std::string cleaned = v;

    // Remove spaces
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ' '), cleaned.end());

    // Replace ',' with '.'
    std::replace(cleaned.begin(), cleaned.end(), ',', '.');

    try
    {
        return std::stod(cleaned); // Convert string to double
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "⛔️Error: Invalid number format: " << v << std::endl;
        return 0.0; // Handle invalid cases
    }
}

// Function to read CSV file
void set_profiled_kernels(std::vector<NcuKernel *> &kernels, const std::string &model_name, int batch_size, std::string data_path = "data/traces")
{
    std::string fullpath = WORKDIR + "/" + data_path + "/nsight-compute/xavier/" + model_name + "_batch-size" + std::to_string(batch_size) + "_nsight-compute.ncu-rep.csv";
    try
    {
        io::CSVReader<24> in(fullpath);
        in.read_header(io::ignore_extra_column,
                       "kernel_name",
                       "block_dim_x",
                       "block_dim_y",
                       "block_dim_z",
                       "grid_dim_x",
                       "grid_dim_y",
                       "grid_dim_z",
                       "register_per_thread",
                       "duration",
                       "static_shared_memory_per_block",
                       "dynamic_shared_memory_per_block",
                       "threads",
                       "waves_per_sm",
                       "shared_memory",
                       "theoretical_occupancy",
                       "theoretical_active_warps_per_SM",
                       "achieved_occupancy",
                       "achieved_active_warps_per_SM",
                       "block_limit_registers",
                       "block_limit_shared_mem",
                       "block_limit_warps",
                       "block_limit_sm",
                       "capability_minor",
                       "capability_major");
        std::string kernel_name;
        int block_dim_x, block_dim_y, block_dim_z, grid_dim_x, grid_dim_y, grid_dim_z, register_per_thread;
        float duration, static_shared_memory_per_block, dynamic_shared_memory_per_block, threads, waves_per_sm, shared_memory, theoretical_occupancy, theoretical_active_warps_per_SM, achieved_occupancy, achieved_active_warps_per_SM, block_limit_registers, block_limit_shared_mem, block_limit_warps, block_limit_sm, capability_minor, capability_major;
        NcuKernel *kernel;
        while (in.read_row(kernel_name,
                           block_dim_x,
                           block_dim_y,
                           block_dim_z,
                           grid_dim_x,
                           grid_dim_y,
                           grid_dim_z,
                           register_per_thread,
                           duration,
                           static_shared_memory_per_block,
                           dynamic_shared_memory_per_block,
                           threads,
                           waves_per_sm,
                           shared_memory,
                           theoretical_occupancy,
                           theoretical_active_warps_per_SM,
                           achieved_occupancy,
                           achieved_active_warps_per_SM,
                           block_limit_registers,
                           block_limit_shared_mem,
                           block_limit_warps,
                           block_limit_sm,
                           capability_minor,
                           capability_major))
        {
            kernel = new NcuKernel();
            kernel->kernel_name = kernel_name;
            kernel->block_dim_x = block_dim_x;
            kernel->block_dim_y = block_dim_y;
            kernel->block_dim_z = block_dim_z;
            kernel->grid_dim_x = grid_dim_x;
            kernel->grid_dim_y = grid_dim_y;
            kernel->grid_dim_z = grid_dim_z;
            kernel->register_per_thread = register_per_thread;
            kernel->duration = duration;
            kernel->static_shared_memory_per_block = static_shared_memory_per_block;
            kernel->dynamic_shared_memory_per_block = dynamic_shared_memory_per_block;
            kernel->threads = threads;
            kernel->waves_per_sm = waves_per_sm;
            kernel->shared_memory = shared_memory;
            kernel->theoretical_occupancy = theoretical_occupancy;
            kernel->theoretical_active_warps_per_SM = theoretical_active_warps_per_SM;
            kernel->achieved_occupancy = achieved_occupancy;
            kernel->achieved_active_warps_per_SM = achieved_active_warps_per_SM;
            kernel->block_limit_registers = block_limit_registers;
            kernel->block_limit_shared_mem = block_limit_shared_mem;
            kernel->block_limit_warps = block_limit_warps;
            kernel->block_limit_sm = block_limit_sm;
            kernel->capability_minor = capability_minor;
            kernel->capability_major = capability_major;
            kernels.push_back(kernel);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "⛔️No trace found at " << fullpath << "\n\t" << e.what() << '\n';
    }
}

void set_memory(map<int, unsigned long> &Memory, const std::string &variant_name, const std::string &hardware_platform, const std::string &data_path = "data/traces")
{
    try
    {
        std::string fullpath = WORKDIR + "/" + data_path + "/mem-pytorch-extracted/" + variant_name + "_mem-pytorch-extracted.csv";
        io::CSVReader<2> in(fullpath);
        in.read_header(io::ignore_extra_column, "batch_size", "total_reserved");
        int batch_size;
        unsigned long total_reserved;
        while (in.read_row(batch_size, total_reserved))
        {
            Memory[batch_size] = fmax(total_reserved, Memory[batch_size]);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "⛔️Error setting memory for " << variant_name << "\n\t" << e.what() << '\n';
    }
}

void set_throughput(map<int, float> &Throughput, const std::string &variant_name, const std::string &hardware_platform, const std::string &data_path = "data/traces")
{
    try
    {
        std::string fullpath = WORKDIR + "/" + data_path + "/inference-time/" + hardware_platform + "/" + variant_name + "-" + hardware_platform + "_inference_time.csv";
        io::CSVReader<2> in(fullpath);
        in.read_header(io::ignore_extra_column, "batch_size", "inference_time");
        int batch_size;
        float inference_time;
        std::map<int, std::vector<float>> Inference_time;
        while (in.read_row(batch_size, inference_time))
        {
            Inference_time[batch_size].push_back(inference_time);
        }
        for (auto &[batch_size, elapsed] : Inference_time)
        {
            Throughput[batch_size] = median(elapsed);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "⛔️Error setting throughput for " << variant_name << "\n\t" << e.what() << '\n';
    }
}

void pre_profiled(Model &model)
{
    for (int batch_size : BATCH_SIZES)
    {
        std::vector<NcuKernel *> *kernels = &((*model.get_Kernel())[batch_size]);
        set_profiled_kernels(*kernels, model.name, batch_size);
    }
    set_memory(*(model.get_Memory()), model.name, model.hardware_platform);
    set_throughput(*(model.get_Throughput()), model.name, model.hardware_platform);
}

#endif // PROFILER_H