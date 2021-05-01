#include "../factor.h"
#include "../influence_diagram.h"
#include "utils.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace Bayes {

	TEST(InfluenceDiagram, WhenXIsNotKnownToD) {
		// simple influence diagram with one factor x for random variable 0
		// and one decision factor d  for decision variable 1 and one utility factor u
		// with parents variables 0 and 1
		// 
		//  x(0)       d(1)
		//    \        /
		//     \      /
		//     `´    `´
		//      u(0,1)
		//

		// one random factor x for variable 0
		Factor x{ {0}, {2}, {7, 3} };
		x.Normalize();

		// one decision factor d with scope 1for decision variable 1
		// for d, we have two possible decision rules. We'll see which is better.
		Factor d0 { {1}, {2}, {1, 0} };
		Factor d1 { {1}, {2}, {0, 1} };
		// vector with all possible decision rules
		std::vector<Factor> all_ds{ d0,d1 };
		// utility factor with parent variables 0 and 1
		Factor u{ {0,1}, {2,2}, {10, 1, 5, 1} };

		// 1. Naive approach: calculate expected utilities for each decision rule
		std::vector<double> all_eus(all_ds.size());
		for (size_t i = 0; i < all_ds.size(); ++i) {
			const auto& decision_rule { all_ds[i] };
			InfluenceDiagram id{ {x},{decision_rule},{u} };
			// calculate expected utility given a decision rule
			all_eus[i] = SimpleCalcExpectedUtility(id);
		}
		std::vector<double> expected{ 7.3000, 3.8000 };
		ExpectVectorElementsNear(all_eus, expected);

		// 2. Calculate expected utility with expected utility factor
		Factor d{ {1}, {2}, {} }; // empty decision rule 
		InfluenceDiagram id{ {x},{d},{u} };
		const auto euf = CalculateExpectedUtilityFactor(id);
		Factor euf_expected{ {1}, {2}, expected };
		ExpectFactorEqual(euf, euf_expected);

		const auto result = OptimizeMEU(id);
		EXPECT_NEAR(result.meu, 7.3, 0.001);
		Factor odr_expected{ {1}, {2}, {1,0} };
		ExpectFactorEqual(result.odr, odr_expected);
	}

	TEST(InfluenceDiagram, WhenX0IsKnownToD) {
		// Uhe decision factor d for decision variable 1 has now a parent 
		// factor x0 for random variable 0. The utility factor u is now a function
		// of the decision d and the factor x2 for random variable 2.
		// 
		//           x0(0)
		//             \
		//              \
		//              `´
		//  x2(2)       d(1,0)
		//    \        /
		//     \      /
		//     `´    `´
		//      u(1,2)

		Factor x0{ {0}, {2}, {7, 3} };
		x0.Normalize();
	
		Factor d{ {1,0}, {2,2}, {1,0,0,1} };
		Factor x2{ {2,0,1}, {2,2,2}, {4, 4, 1, 1, 1, 1, 4, 4} };
		x2 = x2.CPD(2);

		Factor u{ {1,2}, {2,2}, {10, 1, 5, 1} };

		//I3.RandomFactors = [X1 X3];
		//I3.DecisionFactors = D;
		//I3.UtilityFactors = U1;
		InfluenceDiagram i3{ {x0,x2},{d}, {u} };

		// All possible decision rules
		//D1 = D; D2 = D; D3 = D; D4 = D;
		auto d1_0(d);
		auto d1_1(d);
		auto d1_2(d);
		auto d1_3(d);
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