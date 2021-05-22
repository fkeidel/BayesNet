// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#include "gtest/gtest.h"
#include "test_utils.h"
#include "bayesnet/sampling.h"
#include "bayesnet/clique_tree.h"
#include "examples/example_utils.h"
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <random>
#include <chrono>

namespace Bayes
{
	std::pair<uint32_t, uint32_t> Ind2Sub(uint32_t cols, uint32_t i) {
		assert(("cols must not be zero", cols != 0));
		uint32_t row = i / cols;
		uint32_t col = i % cols;
		return { row,col };
	}

	std::pair<ptrdiff_t, ptrdiff_t> operator-(const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
		return {
			static_cast<std::ptrdiff_t>(a.first) - static_cast<std::ptrdiff_t>(b.first),
			static_cast<std::ptrdiff_t>(a.second) - static_cast<std::ptrdiff_t>(b.second)
		};
	}

	std::pair<ptrdiff_t, ptrdiff_t> Abs(const std::pair<ptrdiff_t, ptrdiff_t> p) {
		return { std::abs(p.first), std::abs(p.second) };
	}

	uint32_t Sum(const std::pair<uint32_t, uint32_t> p) {
		return p.first + p.second;
	}

	TEST(Sampling, Ind2Sub) {
		uint32_t n = 4;
		std::vector<std::pair<uint32_t, uint32_t>> sub_expected{
			{0,0},{0,1},{0,2},{0,3},
			{1,0},{1,1},{1,2},{1,3},
			{2,0},{2,1},{2,2},{2,3},
			{3,0},{3,1},{3,2},{3,3}
		};

		std::vector<std::pair<uint32_t, uint32_t>> sub(n * n);
		for (uint32_t i = 0; i < n * n; ++i) {
			sub[i] = Ind2Sub(n, i);
		}
		EXPECT_EQ(sub, sub_expected);
	}

	std::vector<std::vector<uint32_t>> VariableToFactorCorrespondence(const std::vector<uint32_t>& var, std::vector<Factor> factors)
	{
		std::vector<std::vector<uint32_t>> v2f(var.size());
		//V2F = cell(length(V), 1);
		//
		//for f = 1:length(F)
		for (uint32_t f = 0; f < factors.size(); ++f) {
			//    for i = 1:length(F(f).var)
			const auto& factor{ factors[f] };
			for (uint32_t i = 0; i < factor.Var().size(); ++i) {
				//   v = F(f).var(i);
				const auto v = factor.Var(i);
				//   V2F{v} = union(V2F{v}, f);
				v2f[v].push_back(f);
			}//    end
		}//end
		return v2f;
	}

	// Creates a grid Markov Random Field
	// function [toy_network, toy_factors] = ConstructToyNetwork(on_diag_weight, off_diag_weight)
	std::pair<Graph, std::vector<Factor>> CreateGridMrf(uint32_t n, double weight_of_agreement, double weight_of_disagreement)
	{
		//k = 2;   sub-square length
		//V = 1:n*n;
		std::vector<uint32_t> var(n * n, 0);
		std::iota(var.begin(), var.end(), 0);

		//G = struct;
		//G.names = {};
		//for i = 1:length(V)
		//	 G.names{i} = ['pixel', num2str(i)];
		//	 G.card(i) = 2;    
		//end
		std::vector<uint32_t> card(var.size(), 2);

		//edges = zeros(length(V));
		//	cardinality of edges matrix is |v|*|v|
		std::vector<std::vector<uint32_t>> edges(var.size(), std::vector<uint32_t>(var.size(), 0));
		//for i = 1:length(V)
		for (uint32_t i = 0; i < var.size(); ++i) {
			//	 for j = i+1:length(V)
			for (uint32_t j = i + 1; j < var.size(); ++j) {
				//	 Four connected Markov Net
				//	[r_i, c_i] = ind2sub([n,n],i);
				//	[r_j, c_j] = ind2sub([n,n],j);
				const auto row_column_i = Ind2Sub(n, i);
				const auto row_column_j = Ind2Sub(n, j);
				//	if sum(abs([r_i, c_i] - [r_j, c_j])) == 1
				if (Sum(Abs(row_column_i - row_column_j)) == 1) {
					//	edges(i, j) = 1;
					edges[i][j] = 1;
					edges[j][i] = 1; //G.edges = or(edges, edges');
				}
				//	end
			}// end
		}//end
		
		//singleton_factors = [];
		std::vector<Factor> singleton_factors(var.size());
		//for i = 1:length(V)
		for (uint32_t i = 0; i < var.size(); ++i) {
			//	 singleton_factors(i).var = i;
			//	 singleton_factors(i).card = 2;
			//	 if i <= length(V) / 2
			if (i < var.size() / 2) {
				//	singleton_factors(i).val = [0.4, 0.6];
				singleton_factors[i] = { {i},{2},{0.4,0.6} };
			}// else
			else
			{
				//	singleton_factors(i).val = [0.6, 0.4];
				singleton_factors[i] = { {i},{2},{0.6,0.4} };
			}// end
		}// end

		std::vector<uint32_t> find_idx;
		for (uint32_t i = 0; i < var.size(); ++i) {
			for (uint32_t j = 0; j < i; ++j) {
				if (edges[i][j] == 1) {
					find_idx.push_back(i + j * var.size());
				}
			}
		}
		//pairwise_factors = [];
		std::vector<Factor> pairwise_factors;

		//[r, c] = ind2sub([length(V), length(V)], find(edges));
		//edge_list = [r, c];
		std::vector<std::pair<uint32_t, uint32_t>> edge_list(find_idx.size());
		for (uint32_t i = 0; i < find_idx.size(); ++i) {
			edge_list[i] = Ind2Sub(var.size(), find_idx[i]);
		}

		//for i = 1:size(edge_list, 1)
		for (const auto edge : edge_list) {
			//	 pairwise_factors(i).var = edge_list(i, :);
			//	 pairwise_factors(i).card = [2, 2];
			//	 pairwise_factors(i).val = [on_diag_weight, off_diag_weight, ...
			//								off_diag_weight, on_diag_weight];
			pairwise_factors.push_back({ {edge.first, edge.second},{2,2},
				{weight_of_agreement,weight_of_disagreement,weight_of_disagreement,weight_of_agreement} });
		}//end
		
		//F = [singleton_factors, pairwise_factors];
		std::vector<Factor> factors{ singleton_factors };
		factors.insert(factors.end(), pairwise_factors.begin(), pairwise_factors.end());
		//G.var2factors = VariableToFactorCorrespondence(V, F);
		const auto var2factors = VariableToFactorCorrespondence(var, factors);

		//toy_network = G;
		//toy_factors = [singleton_factors, pairwise_factors];
		Graph g{ var, card, edges, var2factors };
		return { {g},{factors} };
	}

	TEST(Sampling, CreateGridMrf) {
		std::vector<std::vector<uint32_t>> edges_expected =
		{
			{0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
			{1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0},
			{0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0},
			{1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0},
			{0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0},
			{0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0},
			{0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0},
			{0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0},
			{0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0},
			{0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0}
		};

		const auto grid_mrf = CreateGridMrf(4, 0.5, 0.5);
		EXPECT_EQ(grid_mrf.first.edges, edges_expected);
	}

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

	// VISUALIZEMCMCMARGINALS
	//
	// This function accepts a list of sample lists, each from a different MCMC run.  It then visualizes
	// the estimated marginals for each variable in V over the lifetime of the MCMC run.
	//
	// samples_list - a list of sample lists; each sample list is a m-by-n matrix where m is the
	// number of samples and n is the number of variables in the state of the Markov chain
	//
	// V - an array of variables
	// D - the dimensions of the variables in V
	// F - a list of factors (used in computing likelihoods of each sample)
	// window_size - size of the window over which to aggregate samples to compute the estimated
	//               marginal at a given time
	// ExactMarginals - the exact marginals of V (optional)
	//
	//function VisualizeMCMCMarginalsFile(plotsdir, samples_list, V, D, F, window_size, ExactMarginals, tname)
	void VisualizeMCMCMarginals();
	//function VisualizeMCMCMarginalsFile(plotsdir, samples_list, V, D, F, window_size, ExactMarginals, tname)
	//
	//for i = 1:length(V)
	//    figure('visible', 'off')
	//    v = V(i);
	//    d = D(i);
	//    title(['Marginal for Variable ', num2str(v)]);
	//    if exist('ExactMarginals') == 1, M = ExactMarginals(i); end;
	//    for j = 1:length(samples_list)
	//        samples_v = samples_list{j}(:, v);
	//        indicators_over_time = zeros(length(samples_v), d);
	//        for k = 1:length(samples_v)
	//            indicators_over_time(k, samples_v(k)) = 1;
	//        end
	//
	//        % estimated_marginal = cumsum(indicators_over_time, 1);
	//        estimated_marginal = [];
	//        for k = 1:size(indicators_over_time, 2)
	//            estimated_marginal = [estimated_marginal, smooth(indicators_over_time(:, k), window_size)];
	//        end
	//        % Prune ends
	//        estimated_marginal = estimated_marginal(window_size/2:end - window_size/2, :);
	//
	//
	//        estimated_marginal = estimated_marginal ./ ...
	//            repmat(sum(estimated_marginal, 2), 1, size(estimated_marginal, 2));
	//        hold on;
	//        plot(estimated_marginal, '-', 'LineWidth', 2);
	//        title(['Est marginals for entry ' num2str(i) ' of samples for ' tname])
	//        if exist('M') == 1
	//            plot([1; size(estimated_marginal, 1)], [M.val; M.val], '--', 'LineWidth', 3);
	//        end
	//        set(gcf,'DefaultAxesColorOrder', rand(d, 3));
	//    end
	//    print('-dpng', [plotsdir, '/MCMC_', tname, '.png']);
	//end
	//
	// Visualize likelihood of sample at each time step
	//all_likelihoods = [];
	//for i = 1:length(samples_list)
	//    samples = samples_list{i};
	//    likelihoods = [];
	//    for j = 1:size(samples, 1)
	//        likelihoods = [likelihoods; LogProbOfJointAssignment(F, samples(j, :))];
	//    end
	//    all_likelihoods = [all_likelihoods, likelihoods];
	//end
	//figure('visible', 'off')
	//plot(all_likelihoods, '-', 'LineWidth', 2);
	//title(['Likelihoods for ' tname])
	//print('-dpng', [plotsdir, '/LIKE_', tname, '.png']);


	//function VisualizeToyImageMarginals(G, M, chain_num, tname)
	void VisualizeToyImageMarginals(Graph graph, std::vector<Factor> marginals, uint32_t chain_num, std::string trans_name, std::string file_path) 
	{
		//n = sqrt(length(G.names));
		const auto n{ std::sqrt(graph.var.size()) };
		//marginal_vector = [];
		std::vector<std::vector<double>> marginal_matrix(n, std::vector<double>(n));
		//for i = 1:length(M)
		for (uint32_t i = 0; i < marginals.size(); ++i) {
			//    marginal_vector(end+1) = M(i).val(1);
			const auto sub{ Ind2Sub(n, i) };
			const auto row{ sub.first };
			const auto col{ sub.second };
			marginal_matrix[row][col] = marginals[i].Val(1); // or 0?
		}//end
		//clims = [0, 1];
		//imagesc(reshape(marginal_vector, n, n), clims);
		//colormap(gray);
		//title(['Marginals for chain ' num2str(chain_num) ' ' tname])
		std::string title{ "Marginals for chain " + std::to_string(chain_num) + " " + trans_name };
		WriteTableToCsv(file_path + trans_name + ".csv", title, marginal_matrix);
	}

	//  Based on scripts by Binesh Bannerjee (saving to file)
	//  and Christian Tott (VisualizeConvergence).
	//  Additionally plots a chart with (possibly inadequate) error score
	//  relative to all exact marginals
	//
	//  These scripts depend on PA4 files for exact inference. Either
	//  copy them to current dir, or add:
	//  addpath '/path/to/your/PA4/Files'
	//
	//  If you notice that your MCMC wanders in circles, it may be
	//  because rand function included with the assignment is still buggy
	//  in your course run. In this case rename rand.m and randi.m
	//  to some other names (like rand.bk and randi.bk). Don't forget to
	//  rename them back if you are going to run test or submit scripts
	//  that depend on them.
	//
	TEST(Sampling, TestToyFile) 
	{
		//rand('seed', 1);
		std::minstd_rand gen;

		// Tunable parameters
		//num_chains_to_run = 3;
		//mix_time = 400;
		//collect = 6000;
		//on_diagonal = 1;
		//off_diagonal = 0.2;
		// 
		uint32_t num_chains_to_run{ 3 };
		uint32_t mix_time{ 400 };
		uint32_t collect{ 6000 };
		double on_diagonal{ 0.3 };
		double off_diagonal{ 1 };
		//
		// Directory to save the plots into, change to your output path
		//plotsdir = './plots_test';
		const std::string file_path{ "c:\\BayesNet\\plots\\" };
		//
		//start = time;
		auto start = std::chrono::system_clock::now();
		//
		// Construct the toy network
		//[toy_network, toy_factors] = ConstructToyNetwork(on_diagonal, off_diagonal);
		auto grid_mrf = CreateGridMrf(4, on_diagonal, off_diagonal);
		const auto& graph{ grid_mrf.first };
		auto& factors{ grid_mrf.second };
		// 
		//toy_evidence = zeros(1, length(toy_network.names));
		//%toy_clique_tree = CreateCliqueTree(toy_factors, []);
		//%toy_cluster_graph = CreateClusterGraph(toy_factors,[]);
		//
		// Exact Inference
		//ExactM = ComputeExactMarginalsBP(toy_factors, toy_evidence, 0);
		const Evidence NO_EVIDENCE;
		const auto exact_marginals = CliqueTreeComputeExactMarginalsBP(factors, NO_EVIDENCE, false);
		
		//graphics_toolkit('gnuplot');
		//figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, ExactM, 1, 'Exact');
		VisualizeToyImageMarginals(graph, exact_marginals, 1, "Exact", file_path);
		//print('-dpng', [plotsdir, '/EXACT.png']);
		//
		// Comment this in to run Approximate Inference on the toy network
		// Approximate Inference
		// % ApproxM = ApproxInference(toy_cluster_graph, toy_factors, toy_evidence);
		//% figure, VisualizeToyImageMarginals(toy_network, ApproxM);
		// ^^ boobytrap, don't uncomment
		//
		// MCMC Inference
		//transition_names = {'Gibbs', 'MHUniform', 'MHGibbs', 'MHSwendsenWang1', 'MHSwendsenWang2'};
		//errors = {};
		//
		//total_cycles = length(transition_names) * num_chains_to_run;
		//cycles_so_far = 0;
		//for j = 1:length(transition_names)
		//    samples_list = {};
		//    errors_list = [];
		//
		//    for i = 1:num_chains_to_run
		//        % Random Initialization
		//        A0 = ceil(rand(1, length(toy_network.names)) .* toy_network.card);
		//
		//        % Initialization to all ones
		//        % A0 = i * ones(1, length(toy_network.names));
		//
		//        MCMCstart = time;
		//        [M, all_samples] = ...
		//            MCMCInference(toy_network, toy_factors, toy_evidence, transition_names{j}, mix_time, collect, 1, A0);
		//        samples_list{i} = all_samples;
		//        disp(['MCMCInference took ', num2str(time-MCMCstart), ' sec.']);
		//        fflush(stdout);
		//        errors_list(:, i) = CalculateErrors(toy_network, ExactM, all_samples, mix_time);
		//	err_start = time;
		//	disp(['Calculating errors took: ', num2str(time-err_start), ' sec.']);
		//	fflush(stdout);
		//
		//        figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, M, i, transition_names{j}); 
		//        print('-dpng', [plotsdir, '/GREY_', transition_names{j}, '_sample', num2str(i), '.png']);
		//
		//        cycles_so_far = cycles_so_far + 1;
		//        cycles_left =  (total_cycles - cycles_so_far);
		//	timeleft = ((time - start) / cycles_so_far) * cycles_left;
		//        disp(['Progress: ', num2str(cycles_so_far), '/', num2str(total_cycles), ...
		//              ', estimated time left to complete: ', num2str(timeleft), ' sec.']);
		//  
		//    end
		//    errors{j} = errors_list;
		//
		//    vis_vars = [3];
		//    VisualizeMCMCMarginalsFile(plotsdir, samples_list, vis_vars, toy_network.card(vis_vars), toy_factors, ...
		//      500, ExactM(vis_vars),transition_names{j});
		//    VisualizeConvergence(plotsdir, samples_list, [3 10], ExactM([3 10]), transition_names{j});
		//
		//    disp(['Saved results for MCMC with transition ', transition_names{j}]);
		//end
		//
		//VisualizeErrors(plotsdir, errors, mix_time, transition_names);
		//
		//elapsed = time - start;
		//
		//fname = [plotsdir, '/report.txt'];
		//file = fopen(fname, 'a');
		//fdisp(file, ['On diag: ', num2str(on_diagonal),
		//             'Off diag: ', num2str(off_diagonal),
		//             'Mix time: ', num2str(mix_time),
		//             'Collect: ', num2str(collect),
		//             'Time consumed: ', num2str(elapsed), ' sec.']);
		//fclose(file);
		//
		//disp(['Done, time consumed: ', num2str(elapsed), ' sec.']);
	}

}