#include "factor.h"
#include "clique_tree.h"
#include <iostream>
#include <iomanip>

// Water Sprinkler
// based on https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/01-tutorial.ipynb.html

using namespace Bayes;

int main()
{
	std::cout << "Water Sprinkler (Clique tree)" << std::endl;

	enum WaterSprinklerVars : uint32_t {
		CLOUDY,
		SPRINKLER,
		RAIN,
		WET_GRAS
	};

	enum DiscreteValues : uint32_t {
		FALSE,
		TRUE
	};

	Factor p_cloudy{ {CLOUDY},   {2},   {0.4,0.6} };
	Factor p_sprinkler_given_cloudy{ {SPRINKLER,CLOUDY}, {2,2}, {0.5,0.5,0.9,0.1} };
	Factor p_rain_given_cloudy{ {RAIN,CLOUDY}, {2,2}, {0.8,0.2,0.2,0.8} };
	Factor p_wet_gras_given_rain_and_sprinkler{ {WET_GRAS,RAIN,SPRINKLER}, {2,2,2}, {1,0,0.1,0.9,0.1,0.9,0.01,0.99} };

	std::vector <Factor> factors{
		{p_cloudy},
		{p_sprinkler_given_cloudy},
		{p_rain_given_cloudy},
		{p_wet_gras_given_rain_and_sprinkler}
	};

	const bool MAP{ false };
	const std::vector<std::pair<uint32_t, uint32_t>> NO_EVIDENCE{};

	// compute all marginals with one run of the Clique Tree algorithm
	const auto marginals = CliqueTreeComputeExactMarginalsBP(factors, NO_EVIDENCE, MAP);

	std::cout << std::fixed << std::setprecision(4);

	auto p = marginals[CLOUDY];
	std::cout << "\n"
		<< "-------------------\n"
		<< "|   P(CLOUDY)     |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p({ FALSE }) << " | " << p({ TRUE }) << " |\n"
		<< "-------------------\n";

	p = marginals[SPRINKLER];
	std::cout << "\n"
		<< "-------------------\n"
		<< "|   P(SPRINKLER)  |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p({ FALSE }) << " | " << p({ TRUE }) << " |\n"
		<< "-------------------\n";

	p = marginals[RAIN];
	std::cout << "\n"
		<< "-------------------\n"
		<< "|   P(RAIN)       |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p({ FALSE }) << " | " << p({ TRUE }) << " |\n"
		<< "-------------------\n";

	p = marginals[WET_GRAS];
	std::cout << "\n"
		<< "-------------------\n"
		<< "|   P(WET_GRAS)   |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p({ FALSE }) << " | " << p({ TRUE }) << " |\n"
		<< "-------------------\n";

	return 0;
}
