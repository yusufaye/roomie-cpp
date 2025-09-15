#ifndef PROFILER_H
#define PROFILER_H

// #include <format>
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "csv.h"
#include "math.h"
#include "general.h"
#include "kernels.h"
#include "datastore.h"
#include "constants.h"

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
    std::string fullpath = WORKDIR + "/" + data_path + "/kernel-profiler/" + model.hardware_platform + "/" + model.name + "_kernel_profiler.json";
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
        spdlog::error("⛔️ Error loading profiled data from {}\n\t{}", fullpath, e.what());
        return;
    }
    Kernel *kernel;
    try
    {
        for (const auto &item : j)
        {
            std::vector<Kernel *> kernels;
            for (const auto &k : item["kernels"])
            {
                kernel = new Kernel();
                kernel->kernel_name = k["kernel_name"];
                kernel->block_dim_x = k["block_dim_x"];
                kernel->block_dim_y = k["block_dim_y"];
                kernel->block_dim_z = k["block_dim_z"];
                kernel->grid_dim_x = k["grid_dim_x"];
                kernel->grid_dim_y = k["grid_dim_y"];
                kernel->grid_dim_z = k["grid_dim_z"];
                kernel->register_per_thread = k["register_per_thread"];
                kernel->duration = k["duration"];
                kernel->shared_memory = k["shared_memory"];
                kernel->achieved_occupancy = k["achieved_occupancy"];
                kernel->achieved_active_warps_per_SM = k["achieved_active_warps_per_SM"];
                kernel->capability_minor = k["capability_minor"];
                kernel->capability_major = k["capability_major"];
                kernels.push_back(kernel);
            }
            model.batch_size = item["batch_size"];
            model.set_kernels(kernels);
        }
    }
    catch (const std::exception &e)
    {
        spdlog::error("Error while loading kernel profil data\n\t{}", e.what());
    }
}

void set_memory(Model &model, const std::string &data_path = "data/traces")
{
    try
    {
        std::string fullpath = WORKDIR + "/" + data_path + "/mem-pytorch-extracted/" + model.name + "_mem-pytorch-extracted.csv";
        io::CSVReader<2> in(fullpath);
        in.read_header(io::ignore_extra_column, "batch_size", "total_reserved");
        int batch_size;
        unsigned long total_reserved;
        while (in.read_row(batch_size, total_reserved))
        {
            model.batch_size = batch_size;
            model.set_memory(fmax(total_reserved, model.get_memory()));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "⛔️ Error setting memory for " << model.name << "\n\t" << e.what() << '\n';
    }
}

void set_throughput(Model &model, const std::string &data_path = "data/traces")
{
    try
    {
        std::string fullpath = WORKDIR + "/" + data_path + "/inference-time/" + model.hardware_platform + "/" + model.name + "-" + model.hardware_platform + "_inference_time.csv";
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
            model.batch_size = batch_size;
            model.set_throughput(batch_size / median(elapsed));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "⛔️ Error setting throughput for " << model.name << "\n\t" << e.what() << '\n';
    }
}

void pre_profiled(Model &model)
{
    set_profiled_kernels(model);
    set_memory(model);
    set_throughput(model);
    for (const auto &[batch_size, kernels] : *model.get_all_kernels())
    {
        model.batch_size = batch_size;
        if (kernels.size() == 0)
        {
            model.batch_size = batch_size;
            model.set_throughput(0.0);
            model.set_memory(0);
        }
    }
}

#endif // PROFILER_H