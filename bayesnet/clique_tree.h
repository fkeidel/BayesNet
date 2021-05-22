// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#ifndef CLIQUE_TREE_H
#define CLIQUE_TREE_H

#include "bayesnet/factor.h"
#include <vector>

namespace Bayes {

	// A clique tree is an undirected tree whose nodes are cliques of variables.
	//
	// The two main attributes are:
	//
	// .clique_list  = list of factors representing the cliques in the tree
	// .clique_edges = adjacency matrix of cliques
	//
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

		const std::vector <Factor>& CliqueList() const { return clique_list_; }
		const Factor& CliqueList(size_t i) const { return clique_list_[i]; }
		const std::vector <std::vector<uint32_t>>& GetCliqueEdges() const { return clique_edges_; }

	private:
		// list of variables in the clique tree
		std::vector<uint32_t> var_;

		// list of factors used in the construction of the clique tree
		std::vector <Factor> factor_list_;

		// adjacency matrix between variables used for Variable Elimination during clique tree creation
		std::vector <std::vector<uint32_t>> variable_edges_; 

		// nodes of the clique tree, each node containing the scope of a clique
		std::vector <std::vector<uint32_t>> nodes_; 

		// indices of intermediate factors created during clique tree creation
		std::vector <ptrdiff_t> message_indices_;

		// adjacency matrix of the cliques
		std::vector <std::vector<uint32_t>> clique_edges_; 

		// list of cliques, where each clique is represented by a factor (potential)
		std::vector <Factor> clique_list_; 
	};

	std::vector<Factor> CliqueTreeComputeExactMarginalsBP(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence, bool is_max);
	std::vector<uint32_t> CliqueTreeMarginalsMaxDecoding(const std::vector<Factor>& m);

}

#endif
