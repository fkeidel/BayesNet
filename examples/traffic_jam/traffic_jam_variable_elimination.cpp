#include "bayesnet/factor.h"
#include "examples/example_utils.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>

// Traffic jam warning system
// --------------------------
// Imagine, you want to implement a Bayesian traffic jam warning system.
// In traffic jams, there are slow vehicles.
// You have two detection systems in your car, to detect slow vehicles.
// A camera and a radar based system, each has a certain sensitivity.
// 
// 
//     T: traffic (jam)
//          |
//          ˅
//     S: slow  (vehicle)
//        /   \
//       /     \
//      ˅       ˅
//  C: camera   R: radar
//  (detection) (detection)
//

using namespace Bayes;

enum TrafficJamVars : uint32_t {
	TRAFFIC,
	SLOW,
	CAMERA,
	RADAR,
	TRAFFIC_T_MINUS_1
};

enum DiscreteValues : uint32_t {
	FALSE,
	TRUE
};

std::vector <Factor> CreateFactorList(
	const double p_traffic,
	const double p_slow_given_traffic,
	const double p_slow_given_no_traffic,
	const double p_camera_given_slow,
	const double p_camera_given_not_slow,
	const double p_radar_given_slow,
	const double p_radar_given_not_slow) {

	Factor pd_traffic{ {TRAFFIC},     {2},   {1 - p_traffic, p_traffic} };
	Factor cpd_slow_given_traffic{ {SLOW,TRAFFIC},{2,2}, {1 - p_slow_given_no_traffic, p_slow_given_no_traffic, 1 - p_slow_given_traffic, p_slow_given_traffic} };
	Factor cpd_camera_given_slow{ {CAMERA,SLOW}, {2,2}, {1 - p_camera_given_not_slow, p_camera_given_not_slow, 1 - p_camera_given_slow,  p_camera_given_slow} };
	Factor cpd_radar_slow{ {RADAR,SLOW},  {2,2}, {1 - p_radar_given_not_slow,  p_radar_given_not_slow,  1 - p_radar_given_slow,   p_radar_given_slow} };

	std::vector <Factor> factors{
		pd_traffic,
		cpd_slow_given_traffic,
		cpd_camera_given_slow,
		cpd_radar_slow
	};

	return factors;
};

std::vector <Factor> CreateFactorListDbn(
	const double p_traffic_minus_1,
	const double p_traffic_t_given_traffic_t_minus_1,
	const double p_slow_given_traffic,
	const double p_slow_given_no_traffic,
	const double p_camera_given_slow,
	const double p_camera_given_not_slow,
	const double p_radar_given_slow,
	const double p_radar_given_not_slow) {

	Factor pd_traffic_minus_1       { {TRAFFIC_T_MINUS_1},{2},{1-p_traffic_minus_1, p_traffic_minus_1} };
	Factor cpd_traffic_t_given_traffic_t_minus_1{ {TRAFFIC,TRAFFIC_T_MINUS_1}, {2,2}, {p_traffic_t_given_traffic_t_minus_1, 1 - p_traffic_t_given_traffic_t_minus_1,
																												  1 - p_traffic_t_given_traffic_t_minus_1, p_traffic_t_given_traffic_t_minus_1} };
	Factor cpd_slow_given_traffic_t { {SLOW,TRAFFIC},  {2,2}, {1 - p_slow_given_no_traffic, p_slow_given_no_traffic, 1 - p_slow_given_traffic, p_slow_given_traffic} };
	Factor cpd_camera_given_slow    { {CAMERA,SLOW},   {2,2}, {1 - p_camera_given_not_slow, p_camera_given_not_slow, 1 - p_camera_given_slow,  p_camera_given_slow} };
	Factor cpd_radar_slow           { {RADAR, SLOW},   {2,2}, {1 - p_radar_given_not_slow,  p_radar_given_not_slow,  1 - p_radar_given_slow,   p_radar_given_slow} };

	std::vector <Factor> factors{
		{pd_traffic_minus_1},
		{cpd_traffic_t_given_traffic_t_minus_1},
		{cpd_slow_given_traffic_t},
		{cpd_camera_given_slow},
		{cpd_radar_slow}
	};

	return factors;
};


int main()
{
	std::cout << "Traffic jam warning system (Variable elimination)" << std::endl;

	double p_traffic = 0.01;
	double p_slow_given_traffic = 0.98;
	double p_slow_given_no_traffic = 0.01;
	double p_camera_given_slow = 0.8;
	double p_camera_given_not_slow = 0.2;
	double p_radar_given_slow = 0.9;
	double p_radar_given_not_slow = 0.1;

	auto f{ CreateFactorList(
		p_traffic, 
		p_slow_given_traffic, 
		p_slow_given_no_traffic, 
		p_camera_given_slow, 
		p_camera_given_not_slow, 
		p_radar_given_slow, 
		p_radar_given_not_slow) };

	// both detection systems detect the slow vehicle
	const Evidence e{ {CAMERA, TRUE}, { RADAR,TRUE } };
	
	// marginal probability of traffic jam given the evidence
	const auto pd_marginal_traffic = VariableEliminationComputeExactMarginalBP(TRAFFIC,f,e);

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n"
		<< "-------------------\n"
		<< "|    P(TRAFFIC)   |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << pd_marginal_traffic({ FALSE }) << " | " << pd_marginal_traffic({ TRUE }) << " |\n"
		<< "-------------------\n\n";

	// time series
	std::cout << "Time series" << std::endl;
	std::cout << "-----------" << std::endl;
	const std::vector<double> p_camera_given_slow_values = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 60, 60, 60, 60, 60, 60, 60, 60, 70, 70, 70, 70, 70, 70, 70, 80, 80, 80,  80,  80,  80,  80,  90,  90,  90,  90,  90,  90, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };
	const std::vector<double> p_radar_given_slow_values =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };

	const auto num_t = p_camera_given_slow_values.size();
	std::vector<double> p_marginal_traffic_values(num_t);
	std::cout << "Static Bayesian Network" << std::endl;
	std::cout << "P(camera|slow), P(radar|slow), P(traffic|camera,radar)" << std::endl;
	for (size_t t = 0; t < p_camera_given_slow_values.size(); ++t) {
		p_camera_given_slow = p_camera_given_slow_values[t] / 100.0;
		p_radar_given_slow = p_radar_given_slow_values[t] / 100.0;
		f = CreateFactorList(
			p_traffic,
			p_slow_given_traffic,
			p_slow_given_no_traffic,
			p_camera_given_slow,
			1 - p_camera_given_slow,
			p_radar_given_slow,
			1 - p_radar_given_slow);
		const Factor pd_marginal_traffic_t = VariableEliminationComputeExactMarginalBP(TRAFFIC, f, e);
		p_marginal_traffic_values[t] = pd_marginal_traffic_t({ TRUE }) * 100.0; // in percent
		std::cout << p_camera_given_slow_values[t] << ", " << p_radar_given_slow_values[t] << ", " << p_marginal_traffic_values[t] << std::endl;
	}

	// write data to file
	std::vector<std::vector<double>> data{ p_camera_given_slow_values, p_radar_given_slow_values, p_marginal_traffic_values };
	WriteTableToCsv(
		"c:\\BayesNet\\examples\\traffic_jam\\traffic.csv", // adapt path for your installation 
		data, "p_camera_given_slow, p_radar_given_slow, pd_marginal_traffic", true
	);

   // Dynamic Bayesian Network (DBN)
	std::cout << "Dynamic Bayesian Network" << std::endl;
	// run with different transition probabilities
	const std::vector<double> p_traffic_transition_values{ 0.5,0.75,0.95 }; 
	for (const auto p_traffic_transition : p_traffic_transition_values) {
		// initialize incoming interface with value from initial time slice
		double p_traffic_minus_1{ pd_marginal_traffic({ TRUE }) };
		std::cout << "P(camera|slow), P(radar|slow), P(traffic|camera,radar)" << std::endl;
		for (size_t t = 0; t < p_camera_given_slow_values.size(); ++t) {
			p_camera_given_slow = p_camera_given_slow_values[t] / 100.0;
			p_radar_given_slow = p_radar_given_slow_values[t] / 100.0;
			f = CreateFactorListDbn(
				p_traffic_minus_1,
				p_traffic_transition,
				p_slow_given_traffic,
				p_slow_given_no_traffic,
				p_camera_given_slow,
				1 - p_camera_given_slow,
				p_radar_given_slow,
				1 - p_radar_given_slow);

			Factor pd_marginal_traffic_t = VariableEliminationComputeExactMarginalBP(TRAFFIC, f, e);
			const double p_marginal_traffic_t = pd_marginal_traffic_t({ TRUE });
			p_marginal_traffic_values[t] = p_marginal_traffic_t * 100.0; // in percent
			p_traffic_minus_1 = p_marginal_traffic_t; // for next iteration
			std::cout << p_camera_given_slow_values[t] << ", " << p_radar_given_slow_values[t] << ", " << p_marginal_traffic_values[t] << std::endl;
		}

		// write data to file
		data = { p_camera_given_slow_values, p_radar_given_slow_values, p_marginal_traffic_values };
		WriteTableToCsv(
			"c:\\BayesNet\\examples\\traffic_jam\\traffic_dbn_" + std::to_string(p_traffic_transition) + ".csv", // adapt path for your installation
			data, "p_camera_given_slow, p_radar_given_slow, pd_marginal_traffic", true
		);
	}

	return 0;
}
