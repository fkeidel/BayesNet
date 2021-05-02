#include "bayesnet/factor.h"
#include "bayesnet/clique_tree.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// Water Sprinkler
// based on https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/01-tutorial.ipynb.html

using namespace Bayes;

int main()
{
	std::cout << "Measure Runtime: Water Sprinkler (Clique tree)" << std::endl;

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

	uint32_t num_runs{ 100 };
	std::vector<double> runtime(num_runs, 0.0);
	std::vector<Factor> marginals{};

	for (uint32_t i = 0; i < num_runs; ++i) {
		std::vector <Factor> f(factors);
		auto t1 = high_resolution_clock::now();
		marginals = CliqueTreeComputeExactMarginalsBP(f, NO_EVIDENCE, MAP);
		auto t2 = high_resolution_clock::now();
		duration<double, std::milli> ms_double = t2 - t1;
		runtime[i] = ms_double.count();
	}

	// get statistics
	std::cout << "num runs: " << num_runs << std::endl;
	std::cout << "runtimes: " << std::endl;
	for (uint32_t i = 0; i < num_runs; ++i) {
		std::cout << "t[" << i << "]: " << runtime[i] << " ms\n";
	}
	const auto min_runtime = *std::min_element(runtime.begin(), runtime.end());
	const auto max_runtime = *std::max_element(runtime.begin(), runtime.end());
	const auto sum_runtime = std::accumulate(runtime.begin(), runtime.end(), 0.0);
	const auto average_runtime = sum_runtime / num_runs;

	// print statistics
	std::cout << std::endl;
	std::cout << "Runtime(min): " << min_runtime << " ms" << std::endl;;
	std::cout << "Runtime(max): " << max_runtime << " ms" << std::endl;;
	std::cout << "Runtime(avg): " << average_runtime << " ms" << std::endl;;

	return 0;
}
