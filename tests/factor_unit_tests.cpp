#include <iostream>

#include "gtest/gtest.h"
#include "../factor.h"

namespace Bayes {
	void ExpectFactorEqual(const Bayes::Factor& result, Bayes::Factor& expected)
	{
		EXPECT_EQ(result.Var(), expected.Var());
		EXPECT_EQ(result.Card(), expected.Card());
		for (size_t i = 0; i < result.Val().size(); ++i) {
			EXPECT_NEAR(result.Val()[i], expected.Val()[i],0.001);
		}
	}

	void ExpectFactorsEqual(const std::vector<Bayes::Factor>& result, std::vector<Bayes::Factor>& expected)
	{
		ASSERT_EQ(result.size(), expected.size());
		for (size_t i = 0; i < expected.size(); ++i) {
			ExpectFactorEqual(result[i], expected[i]);
		}
	}

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

	TEST(FactorProduct, FactorProduct_GivenFactor1IsEmpty_ExpectFactor0) {
		Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		Factor factor1{}; 
		const auto result = FactorProduct(factor0, factor1);
		EXPECT_EQ(result, factor0);
	}

	TEST(FactorProduct, FactorProduct_GivenFactor0IsEmpty_ExpectFactor1) {
		Factor factor0{};
		Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)

		const auto result = FactorProduct(factor0, factor1);
		EXPECT_EQ(result, factor1);
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
		Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		Factor product{ {0,1}, {2,2}, {0.0649, 0.1958, 0.0451, 0.6942} };
		const auto result = FactorProduct(factor0, factor1);
		ExpectFactorEqual(result, product);
	}

	// Factor Marginalization
	// FACTORS.MARGINALIZATION = FactorMarginalization(FACTORS.INPUT(2), [2]);
	// FACTORS.MARGINALIZATION = struct('var', [1], 'card', [2], 'val', [1 1]);
	TEST(FactorMarginalization, FactorMarginalization) {
		Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		Factor marginalization{ {0}, {2}, {1, 1} };
		std::vector<uint32_t> var{ 1 };
		const auto result = FactorMarginalization(factor1, var);
		ExpectFactorEqual(result, marginalization);
	}

	//FACTORS.INPUT(1) contains P(X_1)
	//FACTORS.INPUT(1) = struct('var', [1], 'card', [2], 'val', [0.11, 0.89]);

	//FACTORS.INPUT(2) contains P(X_2 | X_1)
	//FACTORS.INPUT(2) = struct('var', [2, 1], 'card', [2, 2], 'val', [0.59, 0.41, 0.22, 0.78]);

	//FACTORS.INPUT(3) contains P(X_3 | X_2)
	//FACTORS.INPUT(3) = struct('var', [3, 2], 'card', [2, 2], 'val', [0.39, 0.61, 0.06, 0.94]);

	//Observe Evidence
	//FACTORS.EVIDENCE = ObserveEvidence(FACTORS.INPUT, [2 1; 3 2]);
	//FACTORS.EVIDENCE(1) = struct('var', [1], 'card', [2], 'val', [0.11, 0.89]);
	//FACTORS.EVIDENCE(2) = struct('var', [2, 1], 'card', [2, 2], 'val', [0.59, 0, 0.22, 0]);
	//FACTORS.EVIDENCE(3) = struct('var', [3, 2], 'card', [2, 2], 'val', [0, 0.61, 0, 0]);

	TEST(ObserveEvidence, ObserveEvidence) {
		Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		Factor factor2{ {2,1}, {2,2}, {0.39, 0.61, 0.06, 0.94} }; // P(X_3 | X_2)
		std::vector<Factor> f{ factor0,factor1,factor2 };

		Factor expected0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		Factor expected1{ {1,0}, {2,2}, {0.59, 0.0, 0.22, 0.0} }; // P(X_2 | X_1)
		Factor expected2{ {2,1}, {2,2}, {0.0, 0.61, 0.0, 0.0} }; // P(X_3 | X_2)
		std::vector<Factor> expected{ expected0,expected1,expected2 };

		std::vector<std::pair<uint32_t, uint32_t>> e{ {1,0},{2,1} };

		ObserveEvidence(f, e);

		ExpectFactorsEqual(f, expected);
	}

	TEST(VariableElimination, VariableElimination) {

		Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		Factor factor2{ {2,1}, {2,2}, {0.39, 0.61, 0.06, 0.94} }; // P(X_3 | X_2)
		std::vector<Factor> f{ factor0,factor1,factor2 };
		Factor marginal2{ {2}, {2}, {0.1460, 0.8540} }; // P(X_1)

		std::vector<uint32_t> z{ 0, 1 }; // eliminate var 1
		const auto result = VariableElimination(f, z);

		ExpectFactorEqual(result.front(), marginal2);
	}
}

