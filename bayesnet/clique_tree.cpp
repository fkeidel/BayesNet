// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#include "bayesnet/clique_tree.h"
#include "bayesnet/factor.cpp"
#include <unordered_map>
#include <algorithm>

namespace Bayes {

	// CliqueTree(f, evidence) 
	//
	// Takes a list of factors f and evidence e and creates a clique tree. 
	// The values of the cliques are initialized to the initial potentials. 
	// 
	// The clique tree is defined by the following fields:
	// 
	// .clique_list  = list of factors representing the cliques in the tree
	// .clique_edges = adjacency matrix of cliques
	//
	CliqueTree::CliqueTree(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence) 
		: factor_list(f)
	{
		const auto v{ UniqueVars(f) };

		// Set up the adjacency matrix between variables used for Variable Elimination during 
		// clique tree creation (i.e. build an undirected, moralized graph from factors f)
		variable_edges = SetUpAdjacencyMatrix(v,f);

		// run Variable Elimination to build the clique tree
		for (size_t i=0; i < v.size(); ++i)
		{
			// use the Min-Neighbors heuristic to eliminate the variable that has
			// the smallest number of edges connected to it in each cycle
			const auto best_variable = MinNeighbor(variable_edges,v);
			EliminateVar(best_variable);
		}

		// prune the clique tree (absorb small cliques)
		Prune();

      // compute the initial potentials for the cliques
		factor_list = f; // restore original factor list, because factor list 
		                 // has been modified by Variable Elimination
		ObserveEvidence(factor_list, evidence);
		ComputeInitialPotentials();

		return;
	}

	// CliqueTree(nodes, f)
	// 
	// Takes a list of nodes and factors f
	// 
	// Note: You have to call ObserveEvidence() and ComputeInitialPotentials() afterwards
	//       to finish creating the clique tree
	//
	CliqueTree::CliqueTree(std::vector<std::vector<uint32_t>>& nodes, std::vector<Factor>& f)
		: nodes(nodes), factor_list(f) {}
	
	// CliqueTree(clique_list, clique_edges)
	// 
	// Takes in a list of cliques and clique edges
	//   
	CliqueTree::CliqueTree(std::vector<Factor>& clique_list, std::vector <std::vector<uint32_t>>& clique_edges)
		: clique_list(clique_list), clique_edges(clique_edges) {}

	// ComputeInitialPotentials() 
	// 
	// Sets up the cliques in the clique tree
	//
	// Uses the following fields:
	// - nodes:			 nodes in the clique tree, each node containing the scope of a clique
	// - clique_edges: adjacency matrix of the clique_tree
	// - factor_list:  list of factors that were used to build the tree.
	//   
	// Creates the clique_list:
	// - clique_list: represents an array of cliques with appropriate factors 
	//                from factor_list assigned to each clique, where the .val of each 
	//                clique is initialized to the initial potential of that clique
	//
	void CliqueTree::ComputeInitialPotentials() 
	{
		// n = number of cliques
		const auto n{ nodes.size() };

		// First, computes an assignment of factors from factor_list to cliques
		// Then uses that assignment to initialize the cliques in clique_list to 
		// their initial potentials. 

		// initialize variables and cardinalities of the cliques
		clique_list.resize(n);
		for (size_t c = 0; c < n; ++c)
		{
			clique_list[c].SetVar(nodes[c]);
			clique_list[c].SetCard(std::vector<uint32_t>(nodes[c].size(), 0)); // empty cards
		}

		// assign factors to a cliques (alpha is the mapping from factors to cliques)		
		std::vector<size_t> alpha(factor_list.size(), 0);
		// go through all factors
		for (size_t f = 0; f < factor_list.size(); ++f) {
			// go through all cliques
			for (size_t c = 0; c < n; ++c) {
				const auto f_var = factor_list[f].Var();
				const auto c_var = nodes[c];
				// if all variables from f_var are contained in c_var
				SetOperationResult<uint32_t> intersection = Intersection(f_var, c_var);
				if (intersection.values.size() == f_var.size()) { // all f vars contained in c vars
					// map factor to clique
					alpha[f] = c;
					// set cardinalities for variables of the factor in the clique
					const auto f_card = factor_list[f].Card();
					for (size_t v = 0; v < intersection.values.size(); ++v) {
						clique_list[c].SetCard(intersection.right_indices[v], f_card[intersection.left_indices[v]]);
					}
					break;
				}
			}
		}

		// Compute the initial potentials
		for (size_t c = 0; c < n; ++c) 
		{
			const auto& clique{ clique_list[c] };

			// initialize the initial potential psi for the clique with all ones 
			// this is a trick that has the effect that also cliques with only one factor will get sorted by FactorProduct
			std::vector<double> val_all_ones(std::accumulate(clique.Card().begin(), clique.Card().end(), 1, std::multiplies<uint32_t>()), 1.0);
			Factor psi{ clique.Var(),clique.Card(), val_all_ones }; 

			// compute initial potential psi for clique c
			for (size_t f = 0; f < factor_list.size(); ++f) 
			{
				// if factor is contained in clique
				if (alpha[f] == c)
				{ 
					// multiply the factor
					psi = FactorProduct(psi, factor_list[f]);
				}
			}
			assert(("cluster must not be empty", !psi.IsEmpty()));
			clique_list[c] = psi;
		}
		return;
	}

	// EliminateVar(z)
	// 
	// Function used in production of clique trees
	// 
	//	z = variable to eliminate
	//
	void CliqueTree::EliminateVar(uint32_t z)
	{
		std::vector<size_t> use_factors;
		std::vector<uint32_t> scope;

		// go through all factors
		for (size_t f = 0; f < factor_list.size(); ++f) {
			// if factor contains z
			if (std::any_of(factor_list[f].Var().begin(), factor_list[f].Var().end(), [z](uint32_t var) {return var == z; })) {
			   // add factor to factors to multiply
				use_factors.push_back(f);
				// add variables of the factor to the scope of the factors to multiply
				scope = Union(scope, factor_list[f].Var()).values;
			}
		}

		//	update edge map
		//	new edges represent the induced edges for the variable elimination graph.
		for (size_t i = 0; i < scope.size(); ++i) {
			for (size_t j = 0; j < scope.size(); ++j) {
				if (i != j) {
					variable_edges[scope[i]][scope[j]] = 1;
					variable_edges[scope[j]][scope[i]] = 1;
				}
			}
		}

		//	Remove all adjacencies for the variable to be eliminated
		for (size_t j = 0; j < variable_edges[z].size(); ++j) {
			variable_edges[z][j] = 0;
		}
		for (size_t i = 0; i < variable_edges.size(); ++i) {
			variable_edges[i][z] = 0;
		}

		// non_use_factors = list of factors that don't contain z
		std::vector<size_t> range_f(factor_list.size(), 0);
		std::iota(range_f.begin(), range_f.end(), 0);
		std::vector<size_t> non_use_factors = Difference(range_f, use_factors).values;

		std::vector<Factor> new_f(non_use_factors.size());
		std::unordered_map<size_t, size_t> new_map{};

		// copy the non_use_factors to the result
		for (size_t i = 0; i < non_use_factors.size(); ++i) {
			new_f[i] = factor_list[non_use_factors[i]];
			new_map[non_use_factors[i]] = i; // map non_use_factors from f to new_f
		}

		//	Multiply factors which involve z to get a new factor
		Factor new_factor{};
		for (size_t i = 0; i < use_factors.size(); ++i) {
			new_factor = FactorProduct(new_factor, factor_list[use_factors[i]]);
		}

		// eliminate z
		Factor message = new_factor.Marginalize({ z });
		// add message to new factor list
		new_f.push_back(message);
		factor_list = new_f;

		const auto i_new_c{ nodes.size() }; // index of new clique is index of last node + 1
		nodes.push_back(scope); // add node with scope of the new clique
		
		message_indices.push_back(non_use_factors.size());  // message index is inde of last non_use_factor + 1

		// update the edge map of the clique tree																	 
		// first, increase edge map by one row and one column
		// add row with 0s
		clique_edges.push_back(std::vector<uint32_t>(clique_edges.size(), 0)); 
		// add column with 0s
		for (auto& row : clique_edges) 
		{
			row.push_back(0);
		}
		// create link between the current clique and an old clique i, if message from i is used 
		// in construction of the current clique, i.e is contained in the use_factors
		for (size_t i = 0; i < i_new_c; ++i) 
		{
			const auto& it = std::find(use_factors.begin(), use_factors.end(), message_indices[i]);
			if (it != use_factors.end()) 
			{
				clique_edges[i][i_new_c] = 1;
				clique_edges[i_new_c][i] = 1;
				message_indices[i] = -1; // consumed
			}
			else 
			{
				if (message_indices[i] != -1) 
				{
					// remap remaining not used messages to the new order of factors in the next round
					message_indices[i] = new_map[message_indices[i]];
				}
			}
		}
	}

	// Prune()
	// 
	// Prune a clique tree by removing redundant cliques
	//
	// Scan through the neighbors of a clique. If the scope of the clique is
	// contained in the scope of a neighbor, the clique is removed. When removing
	// a clique, all incoming edges are removed and the neighbors are connected.
	// This maintains the running intersection property and gives a more compact 
	// clique tree

	void CliqueTree::Prune() 
	{
		std::vector<size_t> to_remove;
		// check all cliques i, if they can be removed
		for (size_t i = 0; i < nodes.size(); ++i) {
			// if clique i is already removed, skip clique
			const auto& it = std::find(to_remove.begin(), to_remove.end(), i);
			if (it != to_remove.end())
			{
				continue; 
			}
			// get neighbors of clique i from adjacency matrix
			std::vector<size_t> neighbors_i;
			for (size_t j = 0; j < clique_edges.size(); ++j) {
				if (clique_edges[i][j]) {
					neighbors_i.push_back(j);
				}
			}
			// for all neighbors
			for (size_t n = 0; n < neighbors_i.size(); ++n) {
				// get clique index j of neighbor
				const auto j = neighbors_i[n];
				// assert that clique i and neighbor clique are different
				assert(i != j);

				//  if neighbor is already removed, skip neighbour
				const auto& it = std::find(to_remove.begin(), to_remove.end(), j);
				if (it != to_remove.end()) {
					continue;
				}

				//  if scope of clique i is contained in scope of neigbor j
				if (Intersection(nodes[i], nodes[j]).values.size() == nodes[i].size()) 
				{
					// before removing the clique, connect  neighbors
					for (const auto nb : neighbors_i) {
						// if scope of clique i is contained in scope of neighbor nb
						if (Intersection(nodes[i], nodes[nb]).values.size() == nodes[i].size()) 
						{
							const auto other_neighbors = Difference(neighbors_i, { nb }).values;
							// connect neigbors
							for (const auto on : other_neighbors) {
								clique_edges[on][nb] = 1;
							}
							for (const auto d : other_neighbors) {
								clique_edges[nb][d] = 1;
							}
							break;
						}
					}
					// remove the edges for the clique that is to be removed.
					for (size_t e = 0; e < clique_edges[i].size(); ++e) {
						clique_edges[i][e] = 0;
					}
					for (size_t e = 0; e < clique_edges.size(); ++e) {
						clique_edges[e][i] = 0;
					}

					// remove clique
					to_remove.insert(to_remove.begin(), i);
				}
			} 
		}

		// to_keep = list of node to keep (all cliques minus removed cliques)
		std::vector<size_t> range_c_nodes(nodes.size(), 0);
		std::iota(range_c_nodes.begin(), range_c_nodes.end(), 0);
		const auto to_keep = Difference(range_c_nodes, to_remove).values;
		
		// remove nodes for removed cliques
		for (size_t r = 0; r < to_remove.size(); ++r) {
			nodes[to_remove[r]].clear();
		}
		nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [](const auto& node) {return node.empty(); }), nodes.end());

		// build adjacency matrix of cliques to keep
		std::vector < std::vector<uint32_t> > reduced_edges;
		if (!clique_edges.empty()) {
			for (size_t i = 0; i < to_keep.size(); ++i) 
			{
				std::vector<uint32_t> reduced_row;
				for (size_t j = 0; j < to_keep.size(); ++j) 
				{
					reduced_row.push_back(clique_edges[to_keep[i]][to_keep[j]]);
				}
				reduced_edges.push_back(reduced_row);
			}
			clique_edges = reduced_edges;
		}
	}

	// (i,j) = GetNextCliques(message_indices)
	// 
	// Find a pair of cliques ready for message passing
	// 
	// Finds ready cliques in the clique tree given a matrix of current messages. 
	// Returns indices i and j such that clique i is ready to transmit a message to clique j.
	// The method does not return (i,j) if clique i has already passed a message to clique j.
	//
	//	Messages is a n x n matrix of passed messages, where messages(i,j) represents the message 
	// going from clique i to clique j. 
	// If more than one message is ready to be transmitted, the pair (i,j) that is numerically 
	// smallest will be returned.
	// If no such cliques exist, i = j = 0 will returned
	//
	std::pair<uint32_t, uint32_t> CliqueTree::GetNextCliques(std::vector<std::vector<Factor>> messages) 
	{
		const auto n = clique_list.size();
		// for all source cliques
		for (uint32_t i = 0; i < n; ++i) {
			// for all target cliques
			for (uint32_t j = 0; j < n; ++j) {
				// if cliques are connected and message from i to j not yet sent
				if ((clique_edges[i][j] == 1) && (messages[i][j].Var().empty())) {
					// have all incoming messages except from target j been received
					bool all_incoming_received = true;
					for (uint32_t k = 0; k < n; ++k) {
						if ((k != j) && (clique_edges[k][i] == 1) && (messages[k][i].Var().empty())) {
							all_incoming_received = false;
						}
					}
					if (all_incoming_received) {
						// next ready cliques found
						return { i,j };
					}
				}
			}
		}
		// no ready cliques exist
		return { 0,0 };
	}


	// Calibrate() 
	// 
	// Calibrates a given clique tree using sum-product message passing. 
	// 
	// After calling this method, .val for each clique in .clique_list
	// is set to its final calibrated potential.
	//
	void CliqueTree::Calibrate() 
	{
		// Number of cliques in the tree.
		const auto n{ clique_list.size() };

		// Setting up the message matrix
		// messages(i,j) represents the message going from clique i to clique j. 
		std::vector<std::vector<Factor>> messages(n, std::vector<Factor>(n));

		// While there are ready cliques to pass messages between, messages will be sent. 
		// GetNextCliques finds cliques to pass messages between.
		// Once clique i is ready to send message to clique j, the message is computed
		// and put it in message_indices(i,j).
		// There is only one message upward pass and one downward pass

		uint32_t i = 0;
		uint32_t j = 0;
		std::tie(i, j) = GetNextCliques(messages);
		//	 as long as there a messages from a source clique i to target clique j
		while (i || j) {
			// multiply all input messages except from target clique j
			Factor input{};
			for (uint32_t k = 0; k < n; ++k) {
				if (clique_edges[i][k] && (k != j)) {
					input = FactorProduct(input, messages[k][i]);
				}
			} 
			// multiply with own initial potential
			messages[i][j] = FactorProduct(clique_list[i], input);
			// elimiate variables from message from i to j that are not known to target clique j
			const auto eliminate_vars = Difference(clique_list[i].Var(), clique_list[j].Var()).values;
			messages[i][j] = messages[i][j].Marginalize(eliminate_vars);
			// normalize message
			messages[i][j].Normalize();
			// get next message
			std::tie(i, j) = GetNextCliques(messages);
		}

		// Now the clique tree has been calibrated and the final potentials for the cliques 
		// can be computed.
		for (uint32_t i = 0; i < n; ++i) {
			for (uint32_t j = 0; j < n; ++j) {
				if (clique_edges[i][j] == 1) {
					// multiply initial potential with all incoming messages
					clique_list[i] = FactorProduct(clique_list[i], messages[j][i]);
				}
			}
		}   
	}

	// CalibrateMax()
	// 
	// Calibrates a given clique tree using max-sum message passing. 
	// 
	// After calling this method, .val for each clique in .clique_list
	// is set to its final calibrated potential.
	//
	void CliqueTree::CalibrateMax()
	{
		// Number of cliques in the tree.
		const auto n{ clique_list.size() };

		// Setting up the message matrix
		// messages(i,j) represents the message going from clique i to clique j. 
		std::vector<std::vector<Factor>> messages(n, std::vector<Factor>(n));

		// While there are ready cliques to pass messages between, messages will be sent. 
		// GetNextCliques finds cliques to pass messages between.
		// Once clique i is ready to send message to clique j, the message is computed
		// and put it in message_indices(i,j).
		// There is only one message upward pass and one downward pass

		//  convert the values of the cliques to log space
		for (size_t c = 0; c < n; ++c) {
			for (size_t i = 0; i < clique_list[c].Val().size(); ++i) {
				clique_list[c].SetVal(i, std::log(clique_list[c].Val(i)));
			}
		}
		uint32_t i = 0;
		uint32_t j = 0;
		std::tie(i, j) = GetNextCliques(messages);
		//	as long as there a messages from source clique i to target clique j
		while (i || j) {
			// add all input messages except from target clique j
			Factor input{};
			for (uint32_t k = 0; k < n; ++k) {
				if (clique_edges[i][k] && (k != j)) {
					input = FactorSum(input, messages[k][i]);
				}
			}
			// add own initial potential
			messages[i][j] = FactorSum(clique_list[i], input);
			// elimiate variables from message from i to j that are not known to target clique j
			const auto eliminate_vars = Difference(clique_list[i].Var(), clique_list[j].Var()).values;
			messages[i][j] = messages[i][j].MaxMarginalize(eliminate_vars);
			// get next message
			std::tie(i, j) = GetNextCliques(messages);
		}

		// Now the clique tree has been calibrated and the final potentials for the cliques 
		// can be computed.
		for (uint32_t i = 0; i < n; ++i) {
			for (uint32_t j = 0; j < n; ++j) {
				if (clique_edges[i][j] == 1) {
					// sum initial potential with all incoming messages
					clique_list[i] = FactorSum(clique_list[i], messages[j][i]);
				}
			}
		}  
	}

	// CliqueTreeComputeExactMarginalsBP(f,e, is_max)
	// 
	// Runs exact inference and returns the marginals  (if is_max == 0) 
	// or the max-marginals (if is_max == 1) over all the variables
	// 
	// Takes a list of factors f, evidence e, and a flag is_max, runs exact inference 
	// and returns the final marginals for the variables in the network. 
	// If isMax is 1, then it runs exact MAP inference, otherwise exact inference (sum-prod).
	// It returns an array of size equal to the number of variables in the network 
	// where m(i) represents the ith variable and m(i).val represents the marginals 
	// of the ith variable. 
	//
	std::vector<Factor> CliqueTreeComputeExactMarginalsBP(
		std::vector<Factor>& f, 
		const std::vector<std::pair<uint32_t, uint32_t>>& evidence, 
		bool is_max) 
	{
		// create clique tree
		CliqueTree clique_tree{ f, evidence };

		// calibrate
		is_max ? clique_tree.CalibrateMax() : clique_tree.Calibrate();
		
		// get ordered list of all variables
		std::vector<uint32_t> vars{ UniqueVars(clique_tree.CliqueList())};
		
		// calculate marginals for all variables
		std::vector<Factor> m(vars.size());
		// for all variables and cliques
		for (const auto& var : vars) {
			for (const auto& clique : clique_tree.CliqueList()) {
			// if variable is in scope of clique
				if (!Intersection({ var }, clique.Var()).values.empty()) {
					// marginalize all variables except current variable to get marginal
					if (is_max) {
						m[var] = clique.MaxMarginalize(Difference(clique.Var(), { var }).values);
					} // else
					else {
						m[var] = clique.Marginalize(Difference(clique.Var(), { var }).values);
						m[var].Normalize();
					}
					break;
				}
			}
		}
		return m;
	}

	// a = CliqueTreeMarginalsMaxDecoding(m)
	// 
	// Finds the best assignment for each variable from the marginals passed in. 
	// 
	// a(i) is the index of the best instantiation for variable i.
	// 
	// Example: 
	//  two variables {0,1}
	//  marginals for 0 = {0.1, 0.3, 0.6}
	//  marginals for 1 = {0.92, 0.08}
	//  a(0) = 2, a(1) = 0.
	// 
	std::vector<uint32_t> CliqueTreeMarginalsMaxDecoding(const std::vector<Factor>& m) 
	{
		//  Compute the best assignment for variables in the network.
		std::vector<uint32_t> a(m.size(), 0);
		// Iterate through variables
		for (uint32_t i = 0; i < m.size(); ++i) { 
			const auto result = std::max_element(m[i].Val().begin(), m[i].Val().end());
			a[i] = static_cast<uint32_t>(std::distance(m[i].Val().begin(), result));
		}
		return a;
	}

}
	
	
