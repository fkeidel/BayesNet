// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#ifndef GRID_H
#define GRID_H

#include "bayesnet/factor.h"
#include "bayesnet/sampling.h"

namespace Bayes 
{
	std::pair<uint32_t, uint32_t> Ind2Sub(uint32_t cols, uint32_t i);
	std::pair<Graph, std::vector<Factor>> CreateGridMrf(uint32_t n, double weight_of_agreement, double weight_of_disagreement);
}

#endif // GRID_H