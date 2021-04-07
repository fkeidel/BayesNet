#include "../factor.h"
#include "../influence_diagram.h"
#include "utils.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace Bayes {

	TEST(InfluenceDiagram, Test1) {
		/////////////////////////////////////////////////////////////////////////////////
		// Test case 1 - a very simple influence diagram in which X1 is a random variable
		// andD is a decision.The utility U is a function of X1 and D.
		/////////////////////////////////////////////////////////////////////////////////

		//X1 = struct('var', [1], 'card', [2], 'val', [7, 3]);
		Factor x0{ {0}, {2}, {7, 3} };
		//X1.val = X1.val / sum(X1.val);
		const auto sum{ std::accumulate(x0.Val().begin(), x0.Val().end(),0.0) };
		std::vector<double> normalized(x0.Val());
		std::for_each(normalized.begin(), normalized.end(), [sum](double& d) { d /= sum; });
		x0.SetVal(normalized);

		// D = struct('var', [2], 'card', [2], 'val', [1 0]);
		Factor d1{ {1}, {2}, {1, 0} };
		// U1 = struct('var', [1, 2], 'card', [2, 2], 'val', [10, 1, 5, 1]);
		Factor u{ {0,1}, {2,2}, {10, 1, 5, 1} };
		
		//I1.RandomFactors = X1;
		//I1.DecisionFactors = D;
		//I1.UtilityFactors = U1;
		InfluenceDiagram i1{ {x0},{d1}, {u} };

		// All possible decision rules.
		//D1 = D;
		//D2 = D;
		//D2.val = [0 1];% this is a different rule from D1 = [1 0].We_ll see which is better.
		const auto d1_0(d1);
		auto d1_1(d1);
		d1_1.SetVal({ 0, 1 });
		//AllDs = [D1 D2];
		std::vector<Factor> all_ds{ d1_0,d1_1 };

		//allEU = zeros(length(AllDs), 1);
		std::vector<double> all_eus(all_ds.size());
		//for i = 1:length(AllDs)
		for (size_t i = 0; i < all_ds.size(); ++i) {
			//I1.DecisionFactors = AllDs(i);
			i1.decision_factors = { all_ds[i] };
			//allEU(i) = SimpleCalcExpectedUtility(I3);
			all_eus[i] = SimpleCalcExpectedUtility(i1);
		} //end

		// OUTPUT
		// allEU = > [7.3000, 3.8000]
		std::vector<double> expected{ 7.3000, 3.8000 };
		ExpectVectorElementsNear(all_eus, expected);

		// Get EUF...
		const auto euf = CalculateExpectedUtilityFactor(i1);
		// PrintFactor(euf) = >
		// 2
		// 1	7.300000
		// 2	3.800000
		Factor euf_expected{ {1}, {2}, expected };
		ExpectFactorEqual(euf, euf_expected);

		//[meu optdr] = OptimizeMEU(I1)
		const auto result = OptimizeMEU(i1);
		EXPECT_NEAR(result.meu, 7.3, 0.001);
		Factor odr_expected{ {1}, {2}, {1,0} };
		ExpectFactorEqual(result.odr, odr_expected);
		//[meu optdr] = OptimizeWithJointUtility(I1)
		//[meu optdr] = OptimizeLinearExpectations(I1)
		// OUTPUT
		// All should have the same results :
		// meu = > 7.3000
		// PrintFactor(optdr) = >
		// 2     0
		// 1     1
		// 2     0
	}

	TEST(InfluenceDiagram, Test3) {
		/////////////////////////////////////////////////////////////////////////////////
		// Test case 3 - Make D a function of X1.
		/////////////////////////////////////////////////////////////////////////////////
		//X1 = struct('var', [1], 'card', [2], 'val', [7, 3]);
		Factor x0{ {0}, {2}, {7, 3} };
		//X1.val = X1.val / sum(X1.val);
		const auto sum{ std::accumulate(x0.Val().begin(), x0.Val().end(),0.0) };
		std::vector<double> normalized(x0.Val());
		std::for_each(normalized.begin(), normalized.end(), [sum](double& d) { d /= sum; });
		x0.SetVal(normalized);
		// D = struct('var', [2, 1], 'card', [2, 2], 'val', [1, 0, 0, 1]);
		Factor d1{ {1,0}, {2,2}, {1,0,0,1} };
		//X3 = struct('var', [3, 1, 2], 'card', [2, 2, 2], 'val', [4 4 1 1 1 1 4 4]);
		Factor x2{ {2,0,1}, {2,2,2}, {4, 4, 1, 1, 1, 1, 4, 4} };
		//X3 = CPDFromFactor(X3, 3);
		x2 = x2.CPD(2);

		// U is now a function of 3 instead of 2.
		//U1 = struct('var', [2, 3], 'card', [2, 2], 'val', [10, 1, 5, 1]);
		Factor u{ {1,2}, {2,2}, {10, 1, 5, 1} };

		//I3.RandomFactors = [X1 X3];
		//I3.DecisionFactors = D;
		//I3.UtilityFactors = U1;
		InfluenceDiagram i3{ {x0,x2},{d1}, {u} };

		// All possible decision rules
		//D1 = D; D2 = D; D3 = D; D4 = D;
		auto d1_0(d1);
		auto d1_1(d1);
		auto d1_2(d1);
		auto d1_3(d1);
		//D1.val = [1 0 1 0];
		//D2.val = [1 0 0 1];
		//D3.val = [0 1 1 0];
		//D4.val = [0 1 0 1];
		d1_0.SetVal({ 1,0,1,0 });
		d1_1.SetVal({ 1,0,0,1 });
		d1_2.SetVal({ 0,1,1,0 });
		d1_3.SetVal({ 0,1,0,1 });

		//AllDs = [D1 D2 D3 D4];
		std::vector<Factor> all_ds{ d1_0, d1_1, d1_2, d1_3 };

		//allEU = zeros(length(AllDs), 1);
		std::vector<double> all_eus(all_ds.size());
		//for i = 1:length(AllDs)
		for (size_t i = 0; i < all_ds.size(); ++i) {
			//I3.DecisionFactors = AllDs(i);
			i3.decision_factors = { all_ds[i] };
			//allEU(i) = SimpleCalcExpectedUtility(I3);
			all_eus[i] = SimpleCalcExpectedUtility(i3);
		} //end


		// Get EUF...
		// euf = CalculateExpectedUtilityFactor(I3);
		const auto euf = CalculateExpectedUtilityFactor(i3);
		Factor euf_expected{ {0,1}, {2,2}, {5.25,2.25,0.7,0.3} };
		ExpectFactorEqual(euf, euf_expected);

		// PrintFactor(euf) =>
		// 1	2	
		// 1	1	5.250000
		// 2	1	2.250000
		// 1	2	0.700000
		// 2	2	0.300000

		// [meu optdr] = OptimizeMEU(I3)
		const auto result = OptimizeMEU(i3);
		EXPECT_NEAR(result.meu, 7.5, 0.001);
		Factor odr_expected{ {1,0}, {2,2}, { 1,0,1,0} };
		ExpectFactorEqual(result.odr, odr_expected);

		// [meu optdr] = OptimizeWithJointUtility(I3)
		// [meu optdr] = OptimizeLinearExpectations(I3)

		// OUTPUT
		// allEU =
		// 7.5000
		// 5.5500
		// 2.9500
		// 1.0000
		// 
		// meu = 7.5000
		// PrintFactor(optdr) => 
		// 2	1	
		// 1	1	1.000000
		// 2	1	0.000000
		// 1	2	1.000000
		// 2	2	0.000000
	}
}