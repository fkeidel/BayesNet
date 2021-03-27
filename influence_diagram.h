#ifndef INFLUENCE_DIAGRAM_H
#define INFLUENCE_DIAGRAM_H

#include "factor.h"
#include <vector>
#include <utility>

namespace Bayes {

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

