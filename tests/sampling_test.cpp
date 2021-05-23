// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#include "gtest/gtest.h"
#include "test_utils.h"
#include "bayesnet/sampling.h"
#include "bayesnet/grid.h"

namespace Bayes
{

	TEST(Sampling, LogProbOfJointAssignment_GivenOneFactor) {
		const Factor f{ {0,1},{2,2},{30,1,5,10} };
		const std::vector<uint32_t> assignment{ 1,1 };
		const auto log_prob_expected = std::log(10);
		const auto log_prob = LogProbOfJointAssignment({ f }, assignment);
		EXPECT_DOUBLE_EQ(log_prob, log_prob_expected);
	}

	TEST(Sampling, LogProbOfJointAssignment_GivenNetwork) {
		const Factor f0{ {0,1},{2,2},{30,1,5,10} };
		const Factor f1{ {1,2},{2,2},{100,1,1,100} };
		const Factor f2{ {2,3},{2,2},{1,100,100,1} };
		const Factor f3{ {3,0},{2,2},{100,1,1,100} };
		const std::vector<Factor> factors{ f0,f1,f2,f3 };

		const std::vector<uint32_t> assignment{ 1,1,1,1 };
		const auto log_prob_expected = std::log(10) + std::log(100) + std::log(1) + std::log(100);
		const auto log_prob = LogProbOfJointAssignment(factors, assignment);
		EXPECT_DOUBLE_EQ(log_prob, log_prob_expected);
	}

	//      0
	//    /   \
	//   3     1
	//    \   /
	//      2
	TEST(Sampling, BlockLogDistribution) {
		const Factor f0{ {0,1},{2,2},{30,1,5,10} };
		const Factor f1{ {1,2},{2,2},{100,1,1,100} };
		const Factor f2{ {2,3},{2,2},{1,100,100,1} };
		const Factor f3{ {3,0},{2,2},{100,1,1,100} };
		const std::vector<Factor> factors{ f0,f1,f2,f3 };
		std::vector<std::vector<uint32_t>> edges{
			{0,1,0,1},
			{1,0,1,0},
			{0,1,0,1},
			{1,0,1,0}
		};
		std::vector<std::vector<uint32_t>> var2factors{
			{0,3},
			{0,1},
			{1,2},
			{2,3},
		};

		const std::vector<uint32_t> assignment{ 1,1,1,1 };

		Graph g{ {0,1,2,3},{2},edges, var2factors };
		std::vector<double> log_bs_expected{ std::log(5) + std::log(1) , std::log(10) + std::log(100) };
		NormalizeLog(log_bs_expected);

		const auto log_bs = BlockLogDistribution({ 0 }, g, factors, assignment);

		ExpectVectorElementsNear(log_bs, log_bs_expected);
	}

	TEST(Sampling, BlockLogDistribution_WhenGridMrf) {
		const auto grid_mrf = CreateGridMrf(4, 0.5, 0.5);
		const auto& graph{ grid_mrf.first };
		const auto& factors{ grid_mrf.second };

		std::vector<uint32_t> assignment{ 1,	1,	1,	1,	0,	1,	1,	0,	1,	1,	0,	0,	0,	1,	0,	0 };
		std::vector<double> log_bs_expected{ 0, 0.40547 };

		const auto log_bs = BlockLogDistribution({ 0 }, graph, factors, assignment);

		ExpectVectorElementsNear(log_bs, log_bs_expected);
	}

	TEST(Sampling, RandSample_WhenOneValue100SamplesAndNoWeightIncrements_ExpectUniformDistribution) {
		std::vector<uint32_t> vals{ 4 };
		uint32_t num_samples{ 100 };
		std::minstd_rand  gen;

		const auto rand_samples = RandSample(vals, num_samples, true, std::vector<double> {}, gen);
		std::map<int, int> hist;
		for (const auto sample : rand_samples) {
			hist[sample]++;
		}
		for (const auto entry : hist) {
			EXPECT_NEAR(entry.second, 1.0 / 5 * num_samples, 3);
		}
	}

	TEST(Sampling, RandSample_When5Values100SamplesAndNoWeightIncrements_ExpectUniformDistribution) {
		std::vector<uint32_t> vals(5);
		std::iota(vals.begin(), vals.end(), 0);
		uint32_t num_samples{ 100 };
		std::minstd_rand  gen;

		const auto rand_samples = RandSample(vals, num_samples, true, std::vector<double> {}, gen);

		std::map<int, int> hist;
		for (const auto sample : rand_samples) {
			hist[sample]++;
		}
		for (const auto entry : hist) {
			EXPECT_NEAR(entry.second, 1.0 / vals.size() * num_samples, 3);
		}
	}

	TEST(Sampling, RandSample_When5Values100SamplesAndUniformWeightIncrements_ExpectUniformDistribution) {
		std::vector<uint32_t> vals(5);
		std::iota(vals.begin(), vals.end(), 0);
		uint32_t num_samples{ 100 };
		std::minstd_rand  gen;

		const auto rand_samples = RandSample(vals, num_samples, true, std::vector<double> {1.0, 1.0, 1.0, 1.0, 1.0}, gen);

		std::map<int, int> hist;
		for (const auto sample : rand_samples) {
			hist[sample]++;
		}
		for (const auto entry : hist) {
			EXPECT_NEAR(entry.second, 1.0 / vals.size() * num_samples, 3);
		}
	}

	TEST(Sampling, RandSample_When2Values100SamplesAndDifferentWeightIncrements_ExpectDifferentCounts) {
		std::vector<uint32_t> vals(2);
		uint32_t num_samples{ 100 };
		double low_weight{ 1.0 };
		double high_weight{ 4.0 };
		std::iota(vals.begin(), vals.end(), 0);
		std::minstd_rand  gen;

		const auto rand_samples = RandSample(vals, num_samples, true, std::vector<double> {low_weight, high_weight}, gen);

		std::map<int, int> hist;
		for (const auto sample : rand_samples) {
			hist[sample]++;
		}
		EXPECT_NEAR(hist[0], num_samples * low_weight / (low_weight + high_weight), 2);
		EXPECT_NEAR(hist[1], num_samples * high_weight / (low_weight + high_weight), 2);
	}

	TEST(Sampling, RandSample_When10Values10SamplesAndNoReplacement_ExpectAllValuesOnce) {
		std::vector<uint32_t> vals(10);
		std::iota(vals.begin(), vals.end(), 0);
		uint32_t num_samples{ 10 };
		std::minstd_rand  gen;

		const auto rand_samples = RandSample(vals, num_samples, false, std::vector<double> {}, gen);

		std::map<int, int> hist;
		for (const auto sample : rand_samples) {
			hist[sample]++;
		}
		for (const auto entry : hist) {
			EXPECT_EQ(entry.second, 1);
		}
	}

	TEST(Sampling, GibbsTran) {
		const auto grid_mrf = CreateGridMrf(4, 0.5, 0.5);
		const auto& graph{ grid_mrf.first };
		const auto& factors{ grid_mrf.second };

		std::vector<uint32_t> assignment{ 1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0 };
		std::vector<uint32_t> new_assignment_expected{ 0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0 };
		std::minstd_rand  gen;

		const auto new_assignment = GibbsTrans(assignment, graph, factors, gen);

		EXPECT_EQ(new_assignment, new_assignment_expected);
	}

	TEST(Sampling, MHUniformTrans) {
		const auto grid_mrf = CreateGridMrf(4, 0.5, 0.5);
		const auto& graph{ grid_mrf.first };
		const auto& factors{ grid_mrf.second };

		std::vector<uint32_t> assignment{ 1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0 };
		std::vector<uint32_t> new_assignment_expected{ 0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,1 };
		std::minstd_rand gen;

		const auto new_assignment = MHUniformTrans(assignment, graph, factors, gen);

		EXPECT_EQ(new_assignment, new_assignment_expected);
	}

	TEST(Sampling, MHSWTrans_UNIFORM) {
		const auto grid_mrf = CreateGridMrf(4, 0.5, 0.5);
		const auto& graph{ grid_mrf.first };
		const auto& factors{ grid_mrf.second };
		SWVariant variant{ UNIFORM };
		const auto q_list{ CreateQList(graph, factors, variant) };

		std::vector<uint32_t> assignment{ 1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0 };
		std::vector<uint32_t> new_assignment_expected{ 1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0 };

		std::minstd_rand gen;

		const auto new_assignment = MHSWTrans(assignment, graph, factors, gen, variant, q_list);

		EXPECT_EQ(new_assignment, new_assignment_expected);
	}

	TEST(Sampling, MCMCInference) {
		const auto grid_mrf = CreateGridMrf(4, 0.5, 0.5);
		const auto& graph{ grid_mrf.first };
		const auto& factors{ grid_mrf.second };

		std::vector<uint32_t> a0{ 1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0 };
		Evidence evidence;
		Trans trans{ MHUniform };
		uint32_t mix_time{ 0 };
		uint32_t num_samples{ 400 };
		uint32_t sampling_interval{ 1 };
		std::minstd_rand  gen;

		const auto result = MCMCInference(
			graph,
			factors,
			evidence,
			trans,
			mix_time,
			num_samples,
			sampling_interval,
			a0);

		const auto& marginals{ result.first };
		const auto& all_samples{ result.second };

		for (uint32_t i = 0; i < marginals.size(); ++i) {
			const auto& m{ marginals[i] };
			if (i < marginals.size() / 2) {
				ExpectVectorElementsNear(m.Val(), { 0.4,0.6 }, 0.08);
			}
			else {
				ExpectVectorElementsNear(m.Val(), { 0.6,0.4 }, 0.08);
			}
		}
		EXPECT_EQ(all_samples.size(), num_samples + 1);
	}

	TEST(Sampling, CreateAdjacencyMatrixFromQList) {
		QList q_list{
			{1,0,0.5},
			{2,1,0.5},
			{3,2,0.5},
			{5,1,0.5},
			{6,2,0.5},
			{6,5,0.5},
			{9,5,0.5},
			{9,8,0.5},
			{11,7,0.5},
			{11,10,0.5},
			{13,9,0.5},
			{14,10,0.5},
			{15,11,0.5},
			{15,14,0.5}
		};


		std::vector<std::vector<uint32_t>> edges_expected{
			{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0},
			{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0},
			{0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0},
			{0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0}
		};

		std::vector<uint32_t> var(16);
		std::iota(var.begin(), var.end(), 0);

		const auto edges{ CreateAdjacencyMatrixFromQList(q_list, var) };

		EXPECT_EQ(edges, edges_expected);
	}

	TEST(Sampling, DFS)
	{
		std::vector<std::vector<uint32_t>> edges{
			{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0},
			{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0},
			{0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0},
			{0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0}
		};
		std::map<uint32_t, bool> visited_expected{
			{0,true},
			{1,true},
			{2,true},
			{3,true},
			{5,true},
			{6,true},
			{8,true},
			{9,true},
			{13,true},
		};

		std::map<uint32_t, bool> visited;
		DFS({ 5 }, edges, visited);

		EXPECT_EQ(visited, visited_expected);
	}

	TEST(Sampling, FindConnectedComponents)
	{
		std::vector<std::vector<uint32_t>> edges{
			{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0},
			{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0},
			{0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0},
			{0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0}
		};
		std::vector<std::vector<uint32_t>> cc2var_expected{
			{0,1,2,3,5,6,8,9,13},
			{4},
			{7,10,11,14,15},
			{12}
		};

		const auto cc2var{ FindConnectedComponents(edges) };

		EXPECT_EQ(cc2var, cc2var_expected);
	}

}