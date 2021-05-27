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

#endif 
