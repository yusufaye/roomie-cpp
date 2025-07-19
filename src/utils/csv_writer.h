#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include <ctime>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

class CSVWriter
{
public:
  CSVWriter(const std::string &filename) : filename_(filename)
  {
    file.open(filename_, std::ios::out | std::ios::trunc);
    if (!file.is_open())
    {
      std::cerr << "Error opening file: " << filename_ << std::endl;
    }
  }

  ~CSVWriter()
  {
    if (file.is_open())
    {
      file.close();
    }
  }

  template <typename... Args>
  void writeHeader(const std::vector<std::string> &headers)
  {
    if (file.is_open())
    {
      for (size_t i = 0; i < headers.size(); ++i)
      {
        file << headers[i];
        if (i < headers.size() - 1)
        {
          file << ",";
        }
      }
      file << std::endl;
    }
  }

  template <typename... Args>
  void write(const std::tuple<Args...> &data)
  {
    writeImpl(std::index_sequence_for<Args...>{}, data);
  }

private:
  std::string filename_;
  std::ofstream file;

  template <std::size_t... Is, typename... Args>
  void writeImpl(std::index_sequence<Is...>, const std::tuple<Args...> &data)
  {
    if (file.is_open())
    {
      ((file << (Is == 0 ? "" : ",") << std::get<Is>(data)), ...);
      file << std::endl;
    }
  }
};

#endif // CSV_WRITER_H