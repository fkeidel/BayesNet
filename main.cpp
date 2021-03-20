// BayesNet.cpp : Defines the entry point for the application.
//
#include <iostream>
#include "factor.h"

using namespace std;

template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
	out << "[";
	for (size_t i = 0; i < v.size(); ++i) {
		out << v[i];
		if (i < v.size() - 1) {
			out << ",";
		}
	}
	out << "]" << std::endl;
	return out;
}

int main()
{
	cout << "Hello Bayesians!" << endl;
	return 0;
}
