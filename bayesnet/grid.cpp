#include "bayesnet/grid.h"
#include <utility>
#include <vector>

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
}