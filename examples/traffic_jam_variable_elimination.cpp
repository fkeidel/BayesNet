#include "factor.h"
#include <iostream>
#include <iomanip>

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

int main()
{
	std::cout << "Traffic jam warning system (Variable elimination)" << std::endl;

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

	double p_traffic = 0.01;
	double p_slow_given_traffic = 0.98;
	double p_slow_given_no_traffic = 0.01;
	double p_camera_given_slow = 0.8;
	double p_camera_given_not_slow = 0.2;
	double p_radar_given_slow = 0.9;
	double p_radar_given_not_slow = 0.1;

	Factor pd_traffic            { {TRAFFIC},     {2},   {1-p_traffic, p_traffic} };
	Factor cpd_slow_given_traffic{ {SLOW,TRAFFIC},{2,2}, {1-p_slow_given_no_traffic, p_slow_given_no_traffic, 1-p_slow_given_traffic, p_slow_given_traffic} };
	Factor cpd_camera_given_slow { {CAMERA,SLOW}, {2,2}, {1-p_camera_given_not_slow, p_camera_given_not_slow, 1-p_camera_given_slow,  p_camera_given_slow} };
	Factor cpd_radar_slow        { {RADAR,SLOW},  {2,2}, {1-p_radar_given_not_slow,  p_radar_given_not_slow,  1-p_radar_given_slow,   p_radar_given_slow} };

	std::vector <Factor> factors{
		{pd_traffic},
		{cpd_slow_given_traffic},
		{cpd_camera_given_slow},
		{cpd_radar_slow}
	};

	// eliminate all variables except TRAFFIC gives the marginal probability of TRAFFIC
	auto f(factors);
	ObserveEvidence(f, { {CAMERA,TRUE},{RADAR,TRUE} });
	VariableElimination(f, { RADAR, CAMERA, SLOW });
	auto p_marginal_traffic = ComputeJointDistribution(f);
	p_marginal_traffic.Normalize();

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "\n"
		<< "-------------------\n"
		<< "|    P(TRAFFIC)   |\n"
		<< "|-----------------|\n"
		<< "|  FALSE |  TRUE  |\n"
		<< "|--------|--------|\n"
		<< "| " << p_marginal_traffic({ FALSE }) << " | " << p_marginal_traffic({ TRUE }) << " |\n"
		<< "-------------------\n";

	return 0;
}
