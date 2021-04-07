#ifndef CLIQUE_TREE_H
#define CLIQUE_TREE_H

#include "factor.h"
#include <vector>

namespace Bayes {

	struct CliqueTree {
		std::vector < std::vector<uint32_t> > nodes;
		std::vector <uint32_t> card;
		std::vector <Factor> factor_list;
		std::vector <ptrdiff_t> factor_inds;
		std::vector <std::vector<uint32_t> > edges;
		std::vector <Factor> clique_list;
	};

	CliqueTree ComputeInitialPotentials(CliqueTree c);
	CliqueTree CreateCliqueTree(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence);
	void EliminateVar(std::vector<Factor>& f, CliqueTree& c, std::vector<std::vector<uint32_t>>& e, uint32_t z);
	void PruneTree(CliqueTree& c);
	std::pair<uint32_t, uint32_t> GetNextCliques(CliqueTree c, std::vector<std::vector<Factor>> messages);
	void CliqueTreeCalibrate(CliqueTree& c, bool is_max);
	std::vector<Factor> ComputeExactMarginalsBP(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence, bool is_max);
	std::vector<uint32_t> MaxDecoding(const std::vector<Factor>& m);

}

#endif