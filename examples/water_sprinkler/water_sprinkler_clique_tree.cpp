#include "factor.h"
#include "clique_tree.h"
#include <iostream>
#include <iomanip>
#include <map>

// Water Sprinkler
// based on https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/01-tutorial.ipynb.html

using namespace Bayes;

enum WaterSprinklerVars : uint32_t 
{
	CLOUDY,
	SPRINKLER,
	RAIN,
	WET_GRAS
};

std::map<uint32_t, std::string> labels 
{
	{ CLOUDY, "Cloudy" },
	{ SPRINKLER, "Sprinkler" },
	{ RAIN, "Rain" },
	{ WET_GRAS, "Wet gras" }
};

enum DiscreteValues : uint32_t 
{
	FALSE,
	TRUE
};

void PrintMarginal(Factor m) 
{
	const auto& label = labels[m.Var(0)];
	auto n_spaces = 12ULL - label.size();
	n_spaces = std::max(0ULL, n_spaces);
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n"
		<< "-------------------\n"
		<< "|  P(" << label << ")" << std::string(n_spaces,' ') << "|\n"
		<< "|-----------------|\n"
		<< "| FALSE  |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << m({ FALSE }) << " | " << m({ TRUE }) << " |\n"
		<< "-------------------\n";
}

void PrintMarginals(std::vector<Factor> marginals) 
{
	for (const auto& m : marginals) 
	{
		PrintMarginal(m);
	}
}

int main()
{
	std::cout << "Water Sprinkler (Clique tree)" << std::endl;

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
	const auto& marginals = CliqueTreeComputeExactMarginalsBP(factors, NO_EVIDENCE, MAP);

	PrintMarginals(marginals);

	// with evidence
	const std::vector<std::pair<uint32_t, uint32_t>> evidence{ {CLOUDY,0},  {SPRINKLER,1} };
	const auto& marginals_with_evidence = CliqueTreeComputeExactMarginalsBP(factors, evidence, MAP);

	PrintMarginals(marginals_with_evidence);

	return 0;
}
