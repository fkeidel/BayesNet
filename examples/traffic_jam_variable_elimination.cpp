#include "factor.h"
#include "example_utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>

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
	RADAR
};

enum DiscreteValues : uint32_t {
	FALSE,
	TRUE
};

std::vector <Factor> CreateFactorList(const double p_traffic,
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
		{pd_traffic},
		{cpd_slow_given_traffic},
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
	auto p_marginal_traffic = VariableEliminationComputeExactMarginalBP(TRAFFIC,f,e);

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n"
		<< "-------------------\n"
		<< "|    P(TRAFFIC)   |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p_marginal_traffic({ FALSE }) << " | " << p_marginal_traffic({ TRUE }) << " |\n"
		<< "-------------------\n\n";

	// time series
	const std::vector<double> p_camera_given_slow_values = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 60, 60, 60, 60, 60, 60, 60, 60, 70, 70, 70, 70, 70, 70, 70, 80, 80, 80,  80,  80,  80,  80,  90,  90,  90,  90,  90,  90, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };
	const std::vector<double> p_radar_given_slow_values =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };

	std::vector<double> p_marginal_traffic_values(p_camera_given_slow_values.size());
	std::cout << "Time series" << std::endl;
	std::cout << "P(camera|slow), P(radar|slow), P(traffic|camera,radar)" << std::endl;
	for (size_t i = 0; i < p_camera_given_slow_values.size(); ++i) {
		p_camera_given_slow = p_camera_given_slow_values[i] / 100.0;
		p_radar_given_slow = p_radar_given_slow_values[i] / 100.0;
		f = CreateFactorList(
			p_traffic,
			p_slow_given_traffic,
			p_slow_given_no_traffic,
			p_camera_given_slow,
			1 - p_camera_given_slow,
			p_radar_given_slow,
			1 - p_radar_given_slow);
		p_marginal_traffic = VariableEliminationComputeExactMarginalBP(TRAFFIC, f, e);
		p_marginal_traffic_values[i] = p_marginal_traffic({ TRUE }) * 100.0;
		std::cout << p_camera_given_slow_values[i] << ", " << p_radar_given_slow_values[i] << ", " << p_marginal_traffic_values[i] << std::endl;
	}

	// write data to file
	std::vector<std::vector<double>> data{ p_camera_given_slow_values, p_radar_given_slow_values, p_marginal_traffic_values };
	WriteTableToCsv(
		"c:\\BayesNet\\examples\\traffic.csv", // adapt path for your installation
		"p_camera_given_slow, p_radar_given_slow, p_marginal_traffic", 
		data
	);

	return 0;
}
