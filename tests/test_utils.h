#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "bayesnet/factor.h"
#include "gtest/gtest.h"
#include <vector>

namespace Bayes 
{
	template <class T>
	void ExpectVectorElementsNear(std::vector<T> result, std::vector<T> expected, double abs_error = 0.001)
	{
		for (size_t i = 0; i < result.size(); ++i) {
			EXPECT_NEAR(result[i], expected[i], abs_error);
		}
	}

	void ExpectFactorEqual(Factor actual, Factor expected);
	void ExpectFactorsEqual(std::vector <Factor> actual, std::vector <Factor> expected);
}

#endif