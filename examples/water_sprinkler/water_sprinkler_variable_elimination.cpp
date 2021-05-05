#include "bayesnet/factor.h"
#include <iostream>
#include <iomanip>

// Water Sprinkler
// based on https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/01-tutorial.ipynb.html

using namespace Bayes;

int main()
{
	std::cout << "Water Sprinkler (Variable elimination)" << std::endl;

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
		p_cloudy,
		p_sprinkler_given_cloudy,
		p_rain_given_cloudy,
		p_wet_gras_given_rain_and_sprinkler
	};

	// eliminate all variables except WET_GRAS gives the marginal probability of WET_GRAS
	auto f(factors); // make copy to preserve original factors
	VariableElimination(f, {CLOUDY, RAIN, SPRINKLER});
	const auto p_wet_gras = f.front();

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n"
		<< "-------------------\n"
		<< "|   P(WET_GRAS)   |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p_wet_gras({ FALSE }) << " | " << p_wet_gras({ TRUE }) << " |\n"
		<< "-------------------\n";

	// If we want the probabilities of other variables, we have to run Variable Elimination again.
	// To get all marginal probabilities in one run, use the more efficient Clique Tree algorithm.
	f = factors; // restore original factors
	VariableElimination(f, { CLOUDY, SPRINKLER, WET_GRAS });
	const auto p_rain = f.front();

	std::cout << "\n"
		<< "-------------------\n"
		<< "|   P(RAIN)       |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p_rain({ FALSE }) << " | " << p_rain({ TRUE }) << " |\n"
		<< "-------------------\n";

	return 0;
}
