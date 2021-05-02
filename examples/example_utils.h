#ifndef EXAMPLE_UTILS_H
#define EXAMPLE_UTILS_H

#include <fstream>

template<class T>
void WriteTableToCsv(const std::string file_path, std::string header, const std::vector<std::vector<T>>& data) 
{
	std::ofstream file(file_path);
	file << header << std::endl;
	for (size_t t = 0; t < data.front().size(); ++t) {
		for (size_t i = 0; i < data.size(); ++i) {
			file << data[i][t] << (i < (data.size() - 1) ? "," : "");
		}
		file << std::endl;
	}
	file.close();
}

#endif 
