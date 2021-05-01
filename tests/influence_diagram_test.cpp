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
		Factor d{ {1}, {2}, {0,0} }; // dummy decision rule 
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
		// The decision factor d for decision variable 1 has now a parent 
		// factor x0 for random variable 0. The utility factor u is now a function
		// of the decision d and the factor x2 for random variable 2.
		// x2 has parents 0 and 1
		// 
		//       x0(0)
		//      /     \
		//     /       \
		//    `´       `´
		//  x2(2) <--- d(1,0)
		//    \        /
		//     \      /
		//     `´    `´
		//      u(1,2)

		Factor x0{ {0}, {2}, {7, 3} };
		x0.Normalize();
		Factor x2{ {2,0,1}, {2,2,2}, {4, 4, 1, 1, 1, 1, 4, 4} };
		x2 = x2.CPD(2);

		Factor d0{ {1,0}, {2,2}, {1,0,1,0} };
		Factor d1{ {1,0}, {2,2}, {1,0,0,1} };
		Factor d2{ {1,0}, {2,2}, {0,1,1,0} };
		Factor d3{ {1,0}, {2,2}, {0,1,0,1} };
		// vector with all possible decision rules
		std::vector<Factor> all_ds{ d0, d1, d2, d3 };

		Factor u{ {1,2}, {2,2}, {10, 1, 5, 1} };

		// 1. Naive approach: calculate expected utilities for each decision rule
		std::vector<double> all_eus(all_ds.size());
		for (size_t i = 0; i < all_ds.size(); ++i) {
			const auto& decision_rule{ all_ds[i] };
			InfluenceDiagram id{ {x0,x2},{decision_rule}, {u} };
			all_eus[i] = SimpleCalcExpectedUtility(id);
		} 
		std::vector<double> expected{ 7.50, 5.55, 2.95, 1.00 };
		ExpectVectorElementsNear(all_eus, expected);		
	
		// 2. Calculate expected utility with expected utility factor
		Factor d{ {1,0}, {2,2}, {0,0,0,0} }; // dummy decision rule
		InfluenceDiagram id{ {x0,x2},{d}, {u} };

		const auto euf = CalculateExpectedUtilityFactor(id);
		Factor euf_expected{ {0,1}, {2,2}, {5.25,2.25,0.7,0.3} };
		ExpectFactorEqual(euf, euf_expected);

		const auto result = OptimizeMEU(id);
		EXPECT_NEAR(result.meu, 7.5, 0.001);
		Factor odr_expected{ {1,0}, {2,2}, { 1,0,1,0} };
		ExpectFactorEqual(result.odr, odr_expected);
	}
}