#include <iostream>

#include "gtest/gtest.h"
#include "../factor.h"

namespace Bayes {
	TEST(Factor, AssigmentToIndex)
	{
		Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		std::vector<uint32_t> assignment({ 1,0,1 });

		const auto& index = factor.AssigmentToIndex(assignment);

		EXPECT_EQ(index, 5);
	}

	TEST(Factor, IndexToAssignment)
	{
		Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		std::vector<uint32_t> expected_assignment({ 1,0,1 });

		const auto& assignment = factor.IndexToAssignment(5);

		EXPECT_EQ(assignment, expected_assignment);
	}


	TEST(Factor, GetValueOfAssignment)
	{
		Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		std::vector<uint32_t> assignment({ 1,0,1 });
		const auto& value = factor.GetValueOfAssignment(assignment);
		EXPECT_EQ(value, 5);
	}

	TEST(Factor, SetValueOfAssignment)
	{
		Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		std::vector<uint32_t> assignment({ 1,0,1 });
		factor.SetValueOfAssignment(assignment, 8.0);
		const auto& value = factor.GetValueOfAssignment(assignment);
		EXPECT_DOUBLE_EQ(value, 8.0);
	}

	TEST(FactorProduct, FactorProduct_GivenFactor2IsEmpty_ExpectFactor1) {
		Factor factor1{ {1}, {2}, {0.11, 0.89} }; 
		Factor factor2{}; 
		const auto result = FactorProduct(factor1, factor2);
		EXPECT_EQ(result, factor1);
	}

	TEST(FactorProduct, FactorProduct_GivenFactor1IsEmpty_ExpectFactor2) {
		Factor factor1{};
		Factor factor2{ {2,1}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)

		const auto result = FactorProduct(factor1, factor2);
		EXPECT_EQ(result, factor2);
	}

	// FACTORS.INPUT(1) contains P(X_1)
	// FACTORS.INPUT(1) = struct('var', [1], 'card', [2], 'val', [0.11, 0.89]);
	// FACTORS.INPUT(2) contains P(X_2 | X_1)
	// FACTORS.INPUT(2) = struct('var', [2, 1], 'card', [2, 2], 'val', [0.59, 0.41, 0.22, 0.78]);

	// Factor Product
	// FACTORS.PRODUCT = FactorProduct(FACTORS.INPUT(1), FACTORS.INPUT(2));
	// The factor defined here is correct to 4 decimal places.
	// FACTORS.PRODUCT = struct('var', [1, 2], 'card', [2, 2], 'val', [0.0649, 0.1958, 0.0451, 0.6942]);
	TEST(FactorProduct, FactorProduct) {
		Factor factor1{ {1}, {2}, {0.11, 0.89} }; // P(X_1)
		Factor factor2{ {2,1}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		Factor product{ {1,2}, {2,2}, {0.0649, 0.1958, 0.0451, 0.6942} };
		const auto result = FactorProduct(factor1, factor2);
		EXPECT_EQ(result.Var(), product.Var());
		EXPECT_EQ(result.Card(), product.Card());
		for (size_t i = 0; i < result.Val().size(); ++i) {
			EXPECT_DOUBLE_EQ(result.Val()[i], product.Val()[i]);
		}
	}

	// Factor Marginalization
	// FACTORS.MARGINALIZATION = FactorMarginalization(FACTORS.INPUT(2), [2]);
	// FACTORS.MARGINALIZATION = struct('var', [1], 'card', [2], 'val', [1 1]);
	TEST(FactorMarginalization, FactorMarginalization) {
		Factor factor2{ {2,1}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		Factor marginalization{ {1}, {2}, {1, 1} };
		std::vector<uint32_t> var{ 2 };
		const auto result = FactorMarginalization(factor2, var);
		EXPECT_EQ(result.Var(), marginalization.Var());
		EXPECT_EQ(result.Card(), marginalization.Card());
		for (size_t i = 0; i < result.Val().size(); ++i) {
			EXPECT_DOUBLE_EQ(result.Val()[i], marginalization.Val()[i]);
		}
	}

}