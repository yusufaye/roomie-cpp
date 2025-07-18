#ifndef PROFILER_H
#define PROFILER_H

// #include <format>
#include <string>
#include <fstream>
#include <iostream>
#include "csv.h"
#include "math.h"
#include "general.h"
#include "kernels.h"
#include "datastore.h"
#include "constants.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

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
        std::cerr << "⛔️ Error: Invalid number format: " << v << std::endl;
        return 0.0; // Handle invalid cases
    }
}

// Function to read CSV file
void set_profiled_kernels(Model &model, std::string data_path = "data/traces")
{
    std::string fullpath = WORKDIR + "/" + data_path + "/nsight-compute/xavier/" + model.name + "_preprocessed_ncu.json";
    json j;
    try
    {
        // read a JSON file
        std::ifstream i(fullpath);
        i >> j;
        i.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << "⛔️ Error loading profiled data" << "\n\t" << e.what() << '\n';
        return;
    }
    NcuKernel *kernel;
    for (const auto &item : j["traces"])
    {
        std::vector<NcuKernel *> kernels;
        for (const auto &k : item["kernels"])
        {
            kernel = new NcuKernel();
            kernel->kernel_name = k["kernel_name"];
            kernel->block_dim_x = k["block_dim_x"];
            kernel->block_dim_y = k["block_dim_y"];
            kernel->block_dim_z = k["block_dim_z"];
            kernel->grid_dim_x = k["grid_dim_x"];
            kernel->grid_dim_y = k["grid_dim_y"];
            kernel->grid_dim_z = k["grid_dim_z"];
            kernel->register_per_thread = k["register_per_thread"];
            kernel->duration = k["duration"];
            kernel->static_shared_memory_per_block = k["static_shared_memory_per_block"];
            kernel->dynamic_shared_memory_per_block = k["dynamic_shared_memory_per_block"];
            kernel->threads = k["threads"];
            kernel->waves_per_sm = k["waves_per_sm"];
            kernel->shared_memory = k["shared_memory"];
            kernel->theoretical_occupancy = k["theoretical_occupancy"];
            kernel->theoretical_active_warps_per_SM = k["theoretical_active_warps_per_SM"];
            kernel->achieved_occupancy = k["achieved_occupancy"];
            kernel->achieved_active_warps_per_SM = k["achieved_active_warps_per_SM"];
            kernel->block_limit_registers = k["block_limit_registers"];
            kernel->block_limit_shared_mem = k["block_limit_shared_mem"];
            kernel->block_limit_warps = k["block_limit_warps"];
            kernel->block_limit_sm = k["block_limit_sm"];
            kernel->capability_minor = k["capability_minor"];
            kernel->capability_major = k["capability_major"];
            kernels.push_back(kernel);
        }
        (*model.get_Kernel())[item["batch_size"]] = kernels;
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
        std::cerr << "⛔️ Error setting memory for " << variant_name << "\n\t" << e.what() << '\n';
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
        std::cerr << "⛔️ Error setting throughput for " << variant_name << "\n\t" << e.what() << '\n';
    }
}

void pre_profiled(Model &model)
{
    set_profiled_kernels(model);
    set_memory(*(model.get_Memory()), model.name, model.hardware_platform);
    set_throughput(*(model.get_Throughput()), model.name, model.hardware_platform);
}

#endif // PROFILER_H