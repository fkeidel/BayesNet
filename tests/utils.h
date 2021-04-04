#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../factor.h"
#include "gtest/gtest.h"
#include <vector>

namespace Bayes {
	void ExpectVectorElementsNear(std::vector<double> result, std::vector<double> expected);
	void ExpectFactorEqual(Factor actual, Factor expected);
	void ExpectFactorsEqual(std::vector <Factor> actual, std::vector <Factor> expected);
}

#endif