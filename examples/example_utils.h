#ifndef EXAMPLE_UTILS_H
#define EXAMPLE_UTILS_H

#include <fstream>

template<class T>
void WriteTableToCsv(const std::string file_path, const std::vector<std::vector<T>>& data, std::string header = {}, bool transpose = false)
{
	std::ofstream file(file_path);
	if (!header.empty()) {
		file << header << std::endl;
	}
	if (transpose)
	{
		for (size_t i = 0; i < data.front().size(); ++i)
		{
			for (size_t j = 0; j < data.size(); ++j)
			{
				file << data[j][i] << (j < (data.size() - 1) ? "," : "");
			}
			file << std::endl;
		}
		
	}
	else
	{
		for (size_t i = 0; i < data.size(); ++i) 
		{
			for (size_t j = 0; j < data[i].size(); ++j) 
			{
				file << data[i][j] << (j < (data[i].size() - 1) ? "," : "");
			}
			file << std::endl;
		}
		
	}
	file.close();
}

template <class T>
std::vector<std::vector<T>> ReadTableFromCsv(const std::string file_path, bool skip_first_row = false)
{
    std::ifstream file(file_path);
    std::vector<std::vector<T>> result;
    if (file)
    {
        std::string line;
        if (skip_first_row)
        {
            std::getline(file, line);
        }
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::vector<T> row;
            T val;
            while (ss >> val)
            {
                row.push_back(val);
                if (ss.peek() == ',')
                    ss.ignore();
            }
            result.push_back(row);
        }
    }
    else
        std::cout << "Cannot read file: " << file_path << std::endl;
    return result;
}

template <class T>
std::vector<T> GetColumn(const std::vector<std::vector<T>>& data, size_t column)
{
    std::vector<T> result(data.size(), 0.0);
    for (size_t row = 0; row < data.size(); ++row)
    {
        result[row] = data[row][column];
    }
    return result;
}

#endif 
