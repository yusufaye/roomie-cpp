#ifndef PROFILER_H
#define PROFILER_H

// #include <format>
#include <string>
#include <filesystem>
#include <iostream>
#include "NumCpp.hpp"
#include "math.h"
#include "csv.hpp"
#include "kernels.h"
#include "datastore.h"
#include "constants.h"

namespace fs = std::filesystem;

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
        csv::CSVReader reader(fullpath);
        for (csv::CSVRow &row : reader)
        {
            NcuKernel kernel;
            kernel.kernel_name = row["Kernel Name"].get<string>();
            kernel.block_dim_x = convert(row["launch__block_dim_x"].get<string>());
            kernel.block_dim_y = convert(row["launch__block_dim_y"].get<string>());
            kernel.block_dim_z = convert(row["launch__block_dim_z"].get<string>());
            kernel.grid_dim_x = convert(row["launch__grid_dim_x"].get<string>());
            kernel.grid_dim_y = convert(row["launch__grid_dim_y"].get<string>());
            kernel.grid_dim_z = convert(row["launch__grid_dim_z"].get<string>());
            kernel.register_per_thread = convert(row["launch__registers_per_thread"].get<string>());
            kernel.duration = convert(row["gpu__time_duration.sum"].get<string>());
            kernel.static_shared_memory_per_block = convert(row["launch__shared_mem_per_block_static"].get<string>());
            kernel.dynamic_shared_memory_per_block = convert(row["launch__shared_mem_per_block_dynamic"].get<string>());
            kernel.threads = convert(row["launch__thread_count"].get<string>());
            kernel.waves_per_sm = convert(row["launch__waves_per_multiprocessor"].get<string>());
            kernel.shared_memory = convert(row["launch__shared_mem_config_size"].get<string>());
            kernel.theoretical_occupancy = convert(row["sm__maximum_warps_per_active_cycle_pct"].get<string>());
            kernel.theoretical_active_warps_per_SM = convert(row["sm__maximum_warps_avg_per_active_cycle"].get<string>());
            kernel.achieved_occupancy = convert(row["sm__warps_active.avg.pct_of_peak_sustained_active"].get<string>());
            kernel.achieved_active_warps_per_SM = convert(row["sm__warps_active.avg.per_cycle_active"].get<string>());
            kernel.block_limit_registers = convert(row["launch__occupancy_limit_registers"].get<string>());
            kernel.block_limit_shared_mem = convert(row["launch__occupancy_limit_shared_mem"].get<string>());
            kernel.block_limit_warps = convert(row["launch__occupancy_limit_warps"].get<string>());
            kernel.block_limit_sm = convert(row["launch__occupancy_limit_blocks"].get<string>());
            kernel.capability_minor = convert(row["device__attribute_compute_capability_minor"].get<string>());
            kernel.capability_major = convert(row["device__attribute_compute_capability_major"].get<string>());
            kernels.push_back(&kernel);
        }
        // cout << "[INFO] Read a total of " << kernels.size() << " kernels." << endl;
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
        csv::CSVReader reader(fullpath);
        int batch_size;
        for (csv::CSVRow &row : reader)
        {
            batch_size = row["batch_size"].get<int>();
            Memory[batch_size] = fmax(row["total_reserved"].get<unsigned long>(), Memory[batch_size]);
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
        csv::CSVReader reader(fullpath);
        int batch_size;
        map<int, std::vector<float>> inference_time;
        for (csv::CSVRow &row : reader)
        {
            batch_size = row["batch_size"].get<int>();
            inference_time[batch_size].push_back(row["inference_time"].get<float>());
        }
        for (auto &[batch_size, elapsed] : inference_time)
        {
            Throughput[batch_size] = nc::NdArray(elapsed).median().item();
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