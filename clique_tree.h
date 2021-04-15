// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#ifndef CLIQUE_TREE_H
#define CLIQUE_TREE_H

#include "factor.h"
#include <vector>

namespace Bayes {

	class CliqueTree {
	public:
		CliqueTree(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence);
		CliqueTree(std::vector<std::vector<uint32_t>>& nodes, std::vector<Factor>& f);
		CliqueTree(std::vector<Factor>& clique_list, std::vector <std::vector<uint32_t>>& clique_edges);

		void ComputeInitialPotentials();
		void EliminateVar(uint32_t z);
		void Prune();
		std::pair<uint32_t, uint32_t> GetNextCliques(std::vector<std::vector<Factor>> messages);
		void Calibrate();
		void CalibrateMax();

		const std::vector <Factor>& CliqueList() const { return clique_list; }
		const std::vector <std::vector<uint32_t>>& GetCliqueEdges() const { return clique_edges; }

	private:
		std::vector <std::vector<uint32_t>> nodes;
		std::vector <Factor> factor_list;
		std::vector <std::vector<uint32_t>> variable_edges;
		std::vector <std::vector<uint32_t>> clique_edges;
		std::vector <Factor> clique_list;

		std::vector <uint32_t> card; // sorted list of cardinalities of variables in the clique
		std::vector <ptrdiff_t> factor_inds;

	};

	std::vector<Factor> CliqueTreeComputeExactMarginalsBP(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence, bool is_max);
	std::vector<uint32_t> CliqueTreeMarginalsMaxDecoding(const std::vector<Factor>& m);

}

#endif