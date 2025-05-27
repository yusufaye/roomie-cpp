#include "csv.hpp"
#include "kernels.h"

using namespace csv;
using namespace std;




double convert(const std::string& v) {
    std::string cleaned = v;
    
    // Remove spaces
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ' '), cleaned.end());
    
    // Replace ',' with '.'
    std::replace(cleaned.begin(), cleaned.end(), ',', '.');

    try {
        return std::stod(cleaned);  // Convert string to double
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Invalid number format: " << v << std::endl;
        return 0.0; // Handle invalid cases
    }
}

// Function to read CSV file
vector<NcuKernel> get_profiled_data(const string &model_name)
{
    string pathname = "/home/yusufaye/roomie-cpp/src/data/traces/nsight-compute/xavier/" + model_name + "_batch-size1_nsight-compute.ncu-rep.csv";
    csv::CSVReader reader(pathname);
    vector<NcuKernel> kernels;
    for (csv::CSVRow& row: reader) {
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
        kernels.push_back(kernel);
    }
    return kernels;
}