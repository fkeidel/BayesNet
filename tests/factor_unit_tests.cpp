#include <iostream>

#include "gtest/gtest.h"
#include "../factor.h"

namespace Bayes {
	void ExpectFactorEqual(const Bayes::Factor& result, const Bayes::Factor& expected)
	{
		EXPECT_EQ(result.Var(), expected.Var());
		EXPECT_EQ(result.Card(), expected.Card());
		for (size_t i = 0; i < result.Val().size(); ++i) {
			EXPECT_NEAR(result.Val(i), expected.Val(i),0.001);
		}
	}

	void ExpectFactorsEqual(const std::vector<Bayes::Factor>& result, const std::vector<Bayes::Factor>& expected)
	{
		ASSERT_EQ(result.size(), expected.size());
		for (size_t i = 0; i < expected.size(); ++i) {
			ExpectFactorEqual(result[i], expected[i]);
		}
	}

	TEST(Factor, AssigmentToIndex)
	{
		const Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		const std::vector<uint32_t> assignment({ 1,0,1 });

		const auto index = factor.AssigmentToIndex(assignment);

		EXPECT_EQ(index, 5);
	}

	TEST(Factor, IndexToAssignment)
	{
		const Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		const std::vector<uint32_t> expected_assignment({ 1,0,1 });

		const auto assignment = factor.IndexToAssignment(5);

		EXPECT_EQ(assignment, expected_assignment);
	}


	TEST(Factor, GetValueOfAssignment)
	{
		const Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		const std::vector<uint32_t> assignment({ 1,0,1 });
		const auto value = factor.GetValueOfAssignment(assignment);
		EXPECT_EQ(value, 5);
	}

	TEST(Factor, SetValueOfAssignment)
	{
		Factor factor{ {3,1,2}, {2,2,2}, {0,1,2,3,4,5,6,7} };
		const std::vector<uint32_t> assignment({ 1,0,1 });
		factor.SetValueOfAssignment(assignment, 8.0);
		const auto value = factor.GetValueOfAssignment(assignment);
		EXPECT_DOUBLE_EQ(value, 8.0);
	}

	// Factor Marginalize
	// FACTORS.MARGINALIZATION = FactorMarginalization(FACTORS.INPUT(2), [2]);
	// FACTORS.MARGINALIZATION = struct('var', [1], 'card', [2], 'val', [1 1]);
	TEST(Factor, Marginalize) {
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		const Factor marginalization{ {0}, {2}, {1, 1} };
		const std::vector<uint32_t> var{ 1 };
		const auto result = factor1.Marginalize(var);
		ExpectFactorEqual(result, marginalization);
	}

	TEST(Factor, MaxMarginalize) {
		const Factor factor{ {0,6}, {3,2}, {0.0012,0.0183,0.0550,0.0003,0.0122,0.4952} };
		const Factor max_marginalization{ {0}, {3}, {0.0012, 0.0183, 0.4952} };
		const std::vector<uint32_t> var{ 6 };
		const auto result = factor.MaxMarginalize(var);
		ExpectFactorEqual(result, max_marginalization);
	}


	TEST(FactorProduct, FactorProduct_GivenFactor1IsEmpty_ExpectFactor0) {
		const Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor factor1{}; 
		const auto result = FactorProduct(factor0, factor1);
		EXPECT_EQ(result, factor0);
	}

	TEST(FactorProduct, FactorProduct_GivenFactor0IsEmpty_ExpectFactor1) {
		const Factor factor0{};
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)

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
		const Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		const Factor product{ {0,1}, {2,2}, {0.0649, 0.1958, 0.0451, 0.6942} };
		const auto result = FactorProduct(factor0, factor1);
		ExpectFactorEqual(result, product);
	}

	TEST(FactorSum, FactorSum) {
		const Factor factor0{ {0,1}, {2,2}, {3,0,-1,1} }; 
		const Factor factor1{ {1,2}, {2,2}, {4,1.5,0.2,2} };
		const Factor sum{ {0,1,2}, {2,2,2}, {7,4.5,0.2,2,3,0.5,1.2,3} };
		const auto result = FactorSum(factor0, factor1);
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
		const Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		const Factor factor2{ {2,1}, {2,2}, {0.39, 0.61, 0.06, 0.94} }; // P(X_3 | X_2)
		std::vector<Factor> f{ factor0,factor1,factor2 };

		const Factor expected0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor expected1{ {1,0}, {2,2}, {0.59, 0.0, 0.22, 0.0} }; // P(X_2 | X_1)
		const Factor expected2{ {2,1}, {2,2}, {0.0, 0.61, 0.0, 0.0} }; // P(X_3 | X_2)
		const std::vector<Factor> expected{ expected0,expected1,expected2 };

		std::vector<std::pair<uint32_t, uint32_t>> e{ {1,0},{2,1} };

		ObserveEvidence(f, e);

		ExpectFactorsEqual(f, expected);
	}

	TEST(ComputeJointDistribution, ComputeJointDistribution) {

		const Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		const Factor factor2{ {2,1}, {2,2}, {0.39, 0.61, 0.06, 0.94} }; // P(X_3 | X_2)
		const std::vector<Factor> f{ factor0,factor1,factor2 };
		// FACTORS.JOINT = struct('var', [1, 2, 3], 'card', [2, 2, 2], 'val', [0.025311, 0.076362, 0.002706, 0.041652, 0.039589, 0.119438, 0.042394, 0.652548]);
		const Factor joint{ {0,1,2}, {2,2,2}, {0.025311, 0.076362, 0.002706, 0.041652, 0.039589, 0.119438, 0.042394, 0.652548} }; // P(X_1)

		const auto result = ComputeJointDistribution(f);

		ExpectFactorEqual(result, joint);
	}

	TEST(SimpleComputeMarginal, SimpleComputeMarginal) {
		const Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		const Factor factor2{ {2,1}, {2,2}, {0.39, 0.61, 0.06, 0.94} }; // P(X_3 | X_2)
		std::vector<Factor> f{ factor0,factor1,factor2 };
		// FACTORS.MARGINAL = struct('var', [2, 3], 'card', [2, 2], 'val', [0.0858, 0.0468, 0.1342, 0.7332]);
		const Factor marginal{ {1,2}, {2,2}, {0.0858, 0.0468, 0.1342, 0.7332} };

		const std::vector<uint32_t> var{ 1,2 };
		const std::vector<std::pair<uint32_t, uint32_t>> e{ {0,1} };

		//FACTORS.MARGINAL = SimpleComputeMarginal([2, 3], FACTORS.INPUT, [1, 2]);
		const auto result = SimpleComputeMarginal(var, f, e);

		ExpectFactorEqual(result, marginal);
	}

	TEST(VariableElimination, VariableElimination) {
		const Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		const Factor factor2{ {2,1}, {2,2}, {0.39, 0.61, 0.06, 0.94} }; // P(X_3 | X_2)
		std::vector<Factor> f{ factor0,factor1,factor2 };
		const Factor marginal2{ {2}, {2}, {0.1460, 0.8540} }; // P(X_1)

		const std::vector<uint32_t> z{ 0, 1 }; // eliminate var 1
		VariableElimination(f, z);

		ExpectFactorEqual(f.front(), marginal2);
	}

	TEST(UniqueVars, UniqueVars) {
		const Factor factor0{ {0}, {2}, {0.11, 0.89} }; // P(X_1)
		const Factor factor1{ {1,0}, {2,2}, {0.59, 0.41, 0.22, 0.78} }; // P(X_2 | X_1)
		const Factor factor2{ {2,1}, {2,2}, {0.39, 0.61, 0.06, 0.94} }; // P(X_3 | X_2)
		const std::vector<Factor> f{ factor0,factor1,factor2 };

		const std::vector<uint32_t> expected_unique_vars{ 0,1,2 };

		const auto unique_vars{ UniqueVars(f) };

		EXPECT_EQ(unique_vars, expected_unique_vars);
	}

	TEST(NormalizeFactorValue, NormalizeFactorValue) {
		//X1 = struct('var', [1], 'card', [2], 'val', [7, 3]);
		Factor factor0{ {0}, {2}, {7, 3} };
		//X1.val = X1.val / sum(X1.val);

		const Factor factor0_expected{ {0}, {2}, {0.7, 0.3} };

		factor0.Normalize();

		ExpectFactorEqual(factor0, factor0_expected);
	}

	TEST(NormalizeFactorValues, NormalizeFactorValues) {
		//X1 = struct('var', [1], 'card', [2], 'val', [7, 3]);
		const Factor factor0{ {0}, {2}, {7, 3} };
		//X1.val = X1.val / sum(X1.val);
		std::vector<Factor> f{ factor0 };

		const Factor factor0_expected{ {0}, {2}, {0.7, 0.3} };
		const std::vector<Factor> f_expected{ factor0_expected };

		NormalizeFactorValues(f);

		ExpectFactorsEqual(f, f_expected);
	}
}

