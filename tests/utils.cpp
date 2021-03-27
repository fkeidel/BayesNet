#include "utils.h"

namespace Bayes {
	void ExpectVectorElementsNear(std::vector<double> result, std::vector<double> expected)
	{

		for (size_t i = 0; i < result.size(); ++i) {
			EXPECT_NEAR(result[i], expected[i], 0.001);
		}
	}

	void ExpectFactorEqual(Factor actual, Factor expected)
	{
		EXPECT_EQ(actual.Var(), expected.Var());
		EXPECT_EQ(actual.Card(), expected.Card());
		ExpectVectorElementsNear(actual.Val(), expected.Val());
	}
}