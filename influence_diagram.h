#ifndef INFLUENCE_DIAGRAM_H
#define INFLUENCE_DIAGRAM_H

#include "factor.h"
#include <vector>
#include <utility>

namespace Bayes {


	// An InfluenceDiagram is a representation of a decision situation under uncertainty
	// 
	// It has the follwing attributes:
	// 
	// - random_factors =   List of factors for each random variable
	//                      These are the (conditional) probability distributions of each random variable
	// 
	// - decision_factors = List of factors for decision nodes (currently only one supported)
	//                      These are determinstic factors, that tell what action to take for each parent assigment.
	//                      For each parent assignment, they have probability 1 for one action and 0 
	//	                     for all the other actions
	// 
	// - utility_factors  = List of factors representing conditional utilities (currently only one supported)
	//                      Utility factors map each parent assignment to a utility value
	//							   The utility value can be negative (cost).
	//							   .var of the utility factor contains only the parents and no variable for the
	//                      utility factor itself
	//                      Utility factors have no children.
	//
	struct InfluenceDiagram {
		std::vector<Factor> random_factors{};
		std::vector<Factor> decision_factors{};
		std::vector<Factor> utility_factors{};
	};

	struct OptimizeInfluenceDiagramResult {
		double meu{};  // maximum expected utility value
		Factor odr{};  // optimal decision rule
	};

	double SimpleCalcExpectedUtility(const InfluenceDiagram id);
	Factor CalculateExpectedUtilityFactor(const InfluenceDiagram id);
	OptimizeInfluenceDiagramResult OptimizeMEU(InfluenceDiagram i);

}

#endif

