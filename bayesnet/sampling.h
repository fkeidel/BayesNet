// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#ifndef SAMPLING_H
#define SAMPLING_H

#include "bayesnet/factor.h"
#include "bayesnet/utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <map>

namespace Bayes {

	struct Graph {
		std::vector<uint32_t> var;
		std::vector<uint32_t> card;
		std::vector<std::vector<uint32_t>> edges;
		std::vector<std::vector<uint32_t>> var2factors;
	};

	enum Trans {
		Gibbs,
		MHUniform,
		MHGibbs,
		MHSwendsenWang1,
		MHSwendsenWang2
	};

	enum SWVariant {
		UNIFORM,
		BLOCK_SAMPLING
	};

	struct QEntry {
		uint32_t node_i{};
		uint32_t node_j{};
		double q_ij{};
	};

	using QList = std::vector<QEntry>;

	double LogProbOfJointAssignment(const std::vector<Factor>& factors, const std::vector<uint32_t>& assignment);
	std::vector<double> BlockLogDistribution(std::vector<uint32_t> var, Graph graph, std::vector<Factor> factors, std::vector<uint32_t> assignment);
	void NormalizeLog(std::vector<double>& logs);
	std::vector<uint32_t> RandSample(std::vector<uint32_t>& vals, uint32_t num_samp, bool replace, std::vector<double> weight_increments,
		std::minstd_rand& gen);
	std::vector<Factor> ExtractMarginalsFromSamples(const Graph& graph, const std::vector<std::vector<uint32_t>>& samples,
		const std::vector<uint32_t>& collection_indx);

	QList CreateQList(Graph graph, std::vector<Factor> factors, SWVariant variant);
	std::vector <std::vector<uint32_t>> CreateAdjacencyMatrixFromQList(const QList& q_list, const std::vector<uint32_t>& var);
	std::vector<std::vector<uint32_t>> FindConnectedComponents(std::vector<std::vector<uint32_t>> edges);
	void DFS(const uint32_t var, const std::vector<std::vector<uint32_t>> edges, std::map<uint32_t, bool>& visited);

	std::vector<uint32_t> GibbsTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand&  gen);
	std::vector<uint32_t> MHUniformTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand& gen);
	std::vector<uint32_t> MHGibbsTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand& gen);
	std::vector<uint32_t> MHSWTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand& gen, SWVariant variant, const QList& q_list);

	std::pair< std::vector<Factor>, std::vector<std::vector<uint32_t>> > MCMCInference(
		Graph graph,
		std::vector<Factor> factors,
		const Evidence& evidence,
		Trans trans,
		uint32_t mix_time,
		uint32_t num_samples,
		uint32_t sampling_interval,
		std::vector<uint32_t> a0);
}

#endif
