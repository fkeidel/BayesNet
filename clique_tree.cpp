// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#include "clique_tree.h"
#include <factor.cpp>
#include <unordered_map>
#include <algorithm>

namespace Bayes {

	// COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
	// passed in as a parameter.
	//
	//   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
	//   struct with three fields:
	//   - nodes: cell array representing the cliques in the tree.
	//   - edges: represents the adjacency matrix of the tree.
	//   - factorList: represents the list of factors that were used to build
	//   the tree. 
	//   
	//   It returns the standard form of a clique tree P that we will use through 
	//   the rest of the assigment. P is struct with two fields:
	//   - cliqueList: represents an array of cliques with appropriate factors 
	//   from factorList assigned to each clique. Where the .val of each clique
	//   is initialized to the initial potential of that clique.
	//   - edges: represents the adjacency matrix of the tree. 

	// function P = ComputeInitialPotentials(C)
	CliqueTree ComputeInitialPotentials(CliqueTree c) {
		// number of cliques
		//N = length(C.nodes);
		const auto n{ c.nodes.size() };

		//// initialize cluster potentials 
		//P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
		//P.edges = zeros(N);
		std::vector <Factor> clique_list(n);
		//std::vector < std::vector<uint32_t> > edges(n, std::vector<uint32_t>(n));
		//P.edges = C.edges;
		CliqueTree p{};
		p.clique_list = clique_list;
		p.edges = c.edges;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// YOUR CODE HERE
		//
		// First, compute an assignment of factors from factorList to cliques. 
		// Then use that assignment to initialize the cliques in cliqueList to 
		// their initial potentials. 
		// C.nodes is a list of cliques.
		// So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
		// Print out C to get a better understanding of its structure.
		//
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//for i=1:N
		//    P.cliqueList(i).var = C.nodes{i};        
		//end;
		for (size_t i = 0; i < n; ++i) {
			p.clique_list[i].SetVar(c.nodes[i]);
			p.clique_list[i].SetCard(std::vector<uint32_t>(c.nodes[i].size(), 0)); // empty cards
		}

		// Assign factor to a clique
		// alpha=zeros(length(C.factorList),1);			
		std::vector<size_t> alpha(c.factor_list.size(), 0);
		// for k=1:length(C.factorList)
		for (size_t k = 0; k < c.factor_list.size(); ++k) {
			// for i=1:N
			for (size_t i = 0; i < n; ++i) {
				// fVar = C.factorList(k).var;
				const auto f_var = c.factor_list[k].Var();
				// cVar = C.nodes{i};
				const auto c_var = c.nodes[i];
				// if(all(ismember(fVar, cVar)) && alpha(k) == 0)
				// SetOperationResult<uint32_t> intersection = Intersection(f_var, c_var);
				// if (intersection.values.size() == f_var.size()) {
				SetOperationResult<uint32_t> intersection = Intersection(f_var, c_var);
				if (intersection.values.size() == f_var.size()) { // all f vars contained in c vars
					// alpha(k) = i;
					alpha[k] = i;
					// set cardinalities
					const auto f_card = c.factor_list[k].Card();
					for (size_t j = 0; j < intersection.values.size(); ++j) {
						p.clique_list[i].SetCard(intersection.right_indices[j], f_card[intersection.left_indices[j]]);
					}
					break;
				}//        end;
			}//    end;
		} //end;
		//
		// Compute the initial potentials
		//for i=1:N
		for (size_t i = 0; i < n; ++i) {
			//    inds = find(alpha == i);
			//    if(isempty(inds))
			//        P.cliqueList(i).val(:) = 1;
			//        continue;
			//    end;
			//    F1 = C.factorList(inds(1));
			//    for t=2:length(inds)
			//        F2 = C.factorList(inds(t));
			//        F1 = FactorProduct(F1,F2);
			//    end;
			const auto& clique{ p.clique_list[i] };
			std::vector<double> val_all_ones(std::accumulate(clique.Card().begin(), clique.Card().end(), 1, std::multiplies<uint32_t>()), 1.0);
			Factor psi{ clique.Var(),clique.Card(), val_all_ones }; // trick to sort clusters with one factor by FactorProduct
			for (size_t k = 0; k < c.factor_list.size(); ++k) {
				if (alpha[k] == i) { // factor contained in cluster
					psi = FactorProduct(psi, c.factor_list[k]);
				}
			}
			assert(("cluster must not be empty", !psi.IsEmpty()));
			// Not implemented the following variable reindexing, because we assume, the output
			// vars should be sorted, and that's already done by FactorProduct.
			// 
			//    [S, I] = sort(F1.var);
			//    out.card = F1.card(I);
			//    allAssignmentsIn = IndexToAssignment(1:prod(F1.card), F1.card);
			//    allAssignmentsOut = allAssignmentsIn(:,I); // Map from in assgn to out assgn
			//    out.val(AssignmentToIndex(allAssignmentsOut, out.card)) = F1.val;
			//    P.cliqueList(i).card = out.card;
			//    P.cliqueList(i).val = out.val;
			//
			p.clique_list[i] = psi;
		} //end
		return p;
	}

	// CREATECLIQUETREE Takes in a list of factors F, Evidence and returns a 
	// clique tree after calling ComputeInitialPotentials at the end.
	//
	//   P = CREATECLIQUETREE(F, Evidence) Takes a list of factors and creates a clique
	//   tree. The value of the cliques should be initialized to 
	//   the initial potential. 
	//   It returns a clique tree that has the following fields:
	//   - .edges: Contains indices of the nodes that have edges between them.
	//   - .cliqueList: Contains the list of factors used to build the Clique
	//   tree.

	// function P = CreateCliqueTree(F, Evidence)
	CliqueTree CreateCliqueTree(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence) {
		//C.nodes = {};
		CliqueTree c{};
		//V = unique([F(:).var]);
		const auto v{ UniqueVars(f) };
		//
		// Setting up the cardinality for the variables since we only get a list 
		// of factors.

		// C.card = zeros(1, length(V));
		c.card = std::vector<uint32_t>(v.size(), 0);
		//for i = 1 : length(V),
		for (size_t i = 0; i < v.size(); ++i) {
			//	 for j = 1 : length(F)
			for (size_t j = 0; j < f.size(); ++j) {
				const auto& f_var{ f[j].Var() };
				//	if (~isempty(find(F(j).var == i)))
				const auto& it = std::find(f_var.begin(), f_var.end(), i);
				if (it != f_var.end()) {
					const auto ind = std::distance(f_var.begin(), it);
					// C.card(i) = F(j).card(find(F(j).var == i));
					c.card[i] = f[j].Card(ind);
					// break;
					break;
				}//	end
			}//	 end
		}//end

		//C.factorList = F;
		c.factor_list = f;

		// Setting up the adjacency matrix.
		auto edges{ SetUpAdjacencyMatrix(v,f) };
		//
		// cliquesConsidered = 0;
		size_t cliquesConsidered{ 0 };

		// while cliquesConsidered < length(V)
		while (cliquesConsidered < v.size())
		{
			// Using Min-Neighbors where you prefer to eliminate the variable that has
			// the smallest number of edges connected to it. 
			// Everytime you enter the loop, you look at the state of the graph and 
			// pick the variable to be eliminated.
			//
			uint32_t bestClique{ 0 };
			//	bestScore = inf;
			uint32_t bestScore{ std::numeric_limits<uint32_t>::max() };
			//	for i = 1:length(v)
			for (uint32_t i = 0; i < v.size(); ++i) {
				//	score = sum(edges(i, :));
				const auto score = std::accumulate(edges[i].begin(), edges[i].end(), 0U);
				//	if score > 0 && score < bestScore
				if ((score > 0) && (score < bestScore)) {
					//	bestScore = score;
					bestScore = score;
					//	bestClique = i;
					bestClique = i;
				} //  end
			} // end
			// cliquesConsidered = cliquesConsidered + 1;
			++cliquesConsidered;
			// [F, C, edges] = EliminateVar(F, C, edges, bestClique);
			EliminateVar(f, c, edges, bestClique);
		} //end

		// Pruning the tree.
		// C = PruneTree(C);
		PruneTree(c);
		//
		// We are incorporating the effect of evidence in our factor list.
		// (FK) the next lines seem to be wrong in the original code, but it is unsure, if evidence is handled differently when creating clique tree
		// for j = 1:length(Evidence),
		//	 if (Evidence(j) > 0) // ??
		//		C.factorList = ObserveEvidence(C.factorList, [j, Evidence(j)]); 
		ObserveEvidence(c.factor_list, evidence);
		//	 end;
		//end;
		//
		//// Assume that C now has correct cardinality, variables, nodes and edges. 
		//// Here we make the function call to assign factors to cliques and compute the
		//// initial potentials for clusters.
		//
		//P = ComputeInitialPotentials(C);
		const auto p = ComputeInitialPotentials(c);
		//
		//return
		return p;
	}

	// EliminateVar
	// Function used in production of clique trees
	//	F = list of factors
	// 	C = clique creation data
	//	E = adjacency matrix for variables
	//	Z = variable to eliminate

	//function [newF C E] = EliminateVar(F, C, E, Z)
	void EliminateVar(std::vector<Factor>& f, CliqueTree& c, std::vector<std::vector<uint32_t>>& e, uint32_t z)
	{
		//
		//useFactors = [];
		std::vector<size_t> useFactors;

		//scope = [];
		std::vector<uint32_t> scope;

		// for i = 1:length(F)
		for (size_t i = 0; i < f.size(); ++i) {
			//if any(F(i).var == Z)
			if (std::any_of(f[i].Var().begin(), f[i].Var().end(), [z](uint32_t var) {return var == z; })) {
				// useFactors = [useFactors i];
				useFactors.push_back(i);
				// scope = union(scope, F(i).var);
				scope = Union(scope, f[i].Var()).values;
			} // end
		} // end

		//	update edge map
		//	These represent the induced edges for the VE graph.
		//	for i = 1:length(scope)
		for (size_t i = 0; i < scope.size(); ++i) {
			//  for j = 1 : length(scope)
			for (size_t j = 0; j < scope.size(); ++j) {
				//	if i~= j
				if (i != j) {
					//	E(scope(i), scope(j)) = 1;
					//	E(scope(j), scope(i)) = 1;
					e[scope[i]][scope[j]] = 1;
					e[scope[j]][scope[i]] = 1;
				}//	end
			}//	end
		} //end

		//	Remove all adjacencies for the variable to be eliminated
		//	E(Z, :) = 0;
		//  E(:, Z) = 0;
		for (size_t j = 0; j < e[z].size(); ++j) {
			e[z][j] = 0;
		}
		for (size_t i = 0; i < e.size(); ++i) {
			e[i][z] = 0;
		}

		//nonUseFactors = list of factors(not indices!) which are passed through
		//	in this round
		//nonUseFactors = setdiff(1:length(F),[useFactors]);
		std::vector<size_t> range_f(f.size(), 0);
		std::iota(range_f.begin(), range_f.end(), 0);
		std::vector<size_t> nonUseFactors = Difference(range_f, useFactors).values;

		//	newF = list of factors we will return
		std::vector<Factor> newF(nonUseFactors.size());
		//std::vector<size_t> newMap(nonUseFactors.size());
		std::unordered_map<size_t, size_t> newMap{};

		//for i = 1:length(nonUseFactors)
		for (size_t i = 0; i < nonUseFactors.size(); ++i) {
			//	newF(i) = F(nonUseFactors(i));
			newF[i] = f[nonUseFactors[i]];
			//	newmap(nonUseFactors(i)) = i;
			newMap[nonUseFactors[i]] = i; // map nonUseFactors from f to newF
		} //end

		//	Multiply factors which involve Z->newFactor
		//newFactor = struct('var', [], 'card', [], 'val', []);
		Factor newFactor{};
		//for i=1:length(useFactors)
		for (size_t i = 0; i < useFactors.size(); ++i) {
			//	newFactor = FactorProduct(newFactor, F(useFactors(i)));
			newFactor = FactorProduct(newFactor, f[useFactors[i]]);
		} //end

		//	newFactor = FactorMarginalization(newFactor, Z);
		newFactor = FactorMarginalization(newFactor, { z });
		//newF(length(nonUseFactors) + 1) = newFactor;
		newF.push_back(newFactor);
		f = newF;

		//newC = length(C.nodes)+1;
		std::ptrdiff_t newC = static_cast<std::ptrdiff_t>(c.nodes.size()); // index of new cluster
		//C.nodes{newC} = scope;
		c.nodes.push_back(scope);
		//C.factorInds(newC) = length(nonUseFactors)+1;
		c.factor_inds.push_back(nonUseFactors.size());
		// increase edge map
		c.edges.push_back(std::vector<uint32_t>(c.edges.size(), 0)); // add row with 0s
		// add column with 0s
		for (auto& row : c.edges) {
			row.push_back(0);
		}
		// create link if factor used in the computation of another factor
		//for i=1:newC-1 -> for old cliques
		for (ptrdiff_t i = 0; i < newC; ++i) {
			//    if ismember(C.factorInds(i), useFactors)
			// search old factor in current clique
			const auto& it = std::find(useFactors.begin(), useFactors.end(), c.factor_inds[i]);
			if (it != useFactors.end()) {
				// C.edges(i,newC) = 1;
				c.edges[i][newC] = 1;
				// C.edges(newC,i) = 1;
				c.edges[newC][i] = 1;
				// C.factorInds(i) = 0;
				c.factor_inds[i] = -1; // consumed
			} // else
			else {
				// if C.factorInds(i) ~= 0
				if (c.factor_inds[i] != -1) {
					// C.factorInds(i) = newmap(C.factorInds(i));
					c.factor_inds[i] = newMap[c.factor_inds[i]];
				}// end
			}// end
		}//end

	}

	// C = PruneTree(C)
	//
	// Logic:
	// Start with a clique, scan through its neighbors. If you find a neighbor
	// such that it is a superset of the clique you started with, then you know
	// that you can prune the tree. For instance, let's take the following
	// clique tree:
	// ABE -- AB ---AD
	// Let's say we started with AB. We scan through its neighbors and find that
	// AB is a subset of ABE. So we cut off the edges connected to AB and add an
	// edge between ABE and all of AB's other neighbors. This maintains the
	// running intersection property and gives us a more compact clique tree
	// which looks like: ABE -- AD.

	// function C = PruneTree(C)
	void PruneTree(CliqueTree& c) {
		//toRemove = [];
		std::vector<size_t> to_remove;
		//
		//for i=1:length(C.nodes)
		for (size_t i = 0; i < c.nodes.size(); ++i) {
			//    if ismember(i,toRemove), continue, end;
			const auto& it = std::find(to_remove.begin(), to_remove.end(), i);
			if (it != to_remove.end()) {
				continue;
			}
			std::vector<size_t> neighbors_i;
			for (size_t j = 0; j < c.edges.size(); ++j) {
				//  neighborsI = find(C.edges(i,:));
				if (c.edges[i][j]) {
					neighbors_i.push_back(j);
				}
			}
			// for c = 1: length(neighborsI),
			for (size_t n = 0; n < neighbors_i.size(); ++n) {
				//	j = neighborsI(c);
				const auto j = neighbors_i[n];
				//  assert(i ~= j);
				assert(i != j);

				//  if ismember(j,toRemove), continue, end;
				const auto& it = std::find(to_remove.begin(), to_remove.end(), j);
				if (it != to_remove.end()) {
					continue;
				}

				//  if (sum(ismember(C.nodes{i}, C.nodes{j})) == length(C.nodes{i}))
				if (Intersection(c.nodes[i], c.nodes[j]).values.size() == c.nodes[i].size()) {
					// for nk = neighborsI
					for (const auto nk : neighbors_i) {
						// find neighbors and connect with that.
						// if length(intersect(C.nodes{i}, C.nodes{nk})) == length(C.nodes{i})
						if (Intersection(c.nodes[i], c.nodes[nk]).values.size() == c.nodes[i].size()) {
							const auto diff = Difference(neighbors_i, { nk }).values;
							// C.edges(setdiff(neighborsI,[nk]),nk) = 1;
							for (const auto d : diff) {
								c.edges[d][nk] = 1;
							}
							//    C.edges(nk,setdiff(neighborsI,[nk])) = 1;
							for (const auto d : diff) {
								c.edges[nk][d] = 1;
							}
							//    break;
							break;
						}// end
					}// end
					// kill the edges for the clique that is to be removed.

					// C.edges(i,:) = 0;
					// C.edges(:,i) = 0;
					for (size_t e = 0; e < c.edges[i].size(); ++e) {
						c.edges[i][e] = 0;
					}
					for (size_t e = 0; e < c.edges.size(); ++e) {
						c.edges[e][i] = 0;
					}

					// toRemove = [i toRemove];
					to_remove.insert(to_remove.begin(), i);
					//
				}// end
			} // end
		} //end

		//toKeep = setdiff(1:length(C.nodes),toRemove);
		std::vector<size_t> range_c_nodes(c.nodes.size(), 0);
		std::iota(range_c_nodes.begin(), range_c_nodes.end(), 0);

		const auto to_keep = Difference(range_c_nodes, to_remove).values;
		//C.nodes(toRemove) = [];
		for (size_t r = 0; r < to_remove.size(); ++r) {
			c.nodes[to_remove[r]].clear();
		}
		// erase
		c.nodes.erase(std::remove_if(c.nodes.begin(), c.nodes.end(), [](const auto& node) {return node.empty(); }), c.nodes.end());

		// remove all 0 rows/columns
		std::vector < std::vector<uint32_t> > reduced_edges;
		//if isfield(C, 'edges')
		if (!c.edges.empty()) {
			//    C.edges = C.edges(toKeep,toKeep);
			for (size_t i = 0; i < to_keep.size(); ++i) {
				std::vector<uint32_t> reduced_row;
				for (size_t j = 0; j < to_keep.size(); ++j) {
					reduced_row.push_back(c.edges[to_keep[i]][to_keep[j]]);
				}
				reduced_edges.push_back(reduced_row);
			}
			c.edges = reduced_edges;
		}
		//else
		//    C.edges = [];
		//end
		//
	}//end

	//GETNEXTCLIQUES Find a pair of cliques ready for message passing
	//   [i, j] = GETNEXTCLIQUES(P, messages) finds ready cliques in a given
	//   clique tree, P, and a matrix of current messages. Returns indices i and j
	//   such that clique i is ready to transmit a message to clique j.
	//
	//   We are doing clique tree message passing, so
	//   do not return (i,j) if clique i has already passed a message to clique j.
	//
	//	 messages is a n x n matrix of passed messages, where messages(i,j)
	// 	 represents the message going from clique i to clique j. 
	//   This matrix is initialized in CliqueTreeCalibrate as such:
	//      MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
	//
	//   If more than one message is ready to be transmitted, return 
	//   the pair (i,j) that is numerically smallest. If you use an outer
	//   for loop over i and an inner for loop over j, breaking when you find a 
	//   ready pair of cliques, you will get the right answer.
	//
	//   If no such cliques exist, returns i = j = 0.
	//
	//   See also CLIQUETREECALIBRATE
	//
	// based on Coursera Course PGM by Daphne Koller, Stanford University, 2012

	//function [i, j] = GetNextCliques(P, messages)
	std::pair<uint32_t, uint32_t> GetNextCliques(CliqueTree c, std::vector<std::vector<Factor>> messages) {
		// initialization
		// you should set them to the correct values in your code
		//i = 0;
		//j = 0;

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// YOUR CODE HERE
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// for all messages from cluster i to j
		// N = size(messages, 1);
		const auto n = c.clique_list.size();
		//for i = 1:N (source)
		for (uint32_t i = 0; i < n; ++i) {
			// for j = 1:N (target)
			for (uint32_t j = 0; j < n; ++j) {
				// if P.edges(i, j) == 1 && isempty(messages(i, j).var)
				// is edge connected and empty ?
				if ((c.edges[i][j] == 1) && (messages[i][j].Var().empty())) {
					// have all incoming messages except from target  j been received
					bool all_incoming_received = true;
					for (uint32_t k = 0; k < n; ++k) {
						if ((k != j) && (c.edges[k][i] == 1) && (messages[k][i].Var().empty())) {
							all_incoming_received = false;
						}
					}
					if (all_incoming_received) {
						return { i,j };
					}
				}
			}
		}
		// no clique exists
		return { 0,0 };
	}//end


	// CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
	// clique tree calibration.

	//   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
	//   according to the value of isMax flag. If isMax is 1, it uses max-sum
	//   message passing, otherwise uses sum-product. This function 
	//   returns the clique tree where the .val for each clique in .cliqueList
	//   is set to the final calibrated potentials.

	//function P = CliqueTreeCalibrate(P, isMax)
	void CliqueTreeCalibrate(CliqueTree& c, bool is_max) {

		// YOUR CODE HERE
		// Number of cliques in the tree.
		// N = length(P.cliqueList);
		const auto n{ c.clique_list.size() };

		// Setting up the messages that will be passed.
		// MESSAGES(i,j) represents the message going from clique i to clique j. 
		// MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
		std::vector<std::vector<Factor>> messages(n, std::vector<Factor>(n));

		// While there are ready cliques to pass messages between, keep passing
		// messages. Use GetNextCliques to find cliques to pass messages between.
		// Once you have clique i that is ready to send message to clique
		// j, compute the message and put it in MESSAGES(i,j).
		// Remember that you only need an upward pass and a downward pass.
		//
		// 
		//if isMax
		if (is_max) {
			//  convert to log space
			//	 for i = 1:N       
			for (size_t i = 0; i < n; ++i) {
				for (size_t j = 0; j < c.clique_list[i].Val().size(); ++j) {
					// P.cliqueList(i).val = log(P.cliqueList(i).val);
					c.clique_list[i].SetVal(j, std::log(c.clique_list[i].Val(j)));
				}
			} //	 end
			uint32_t i = 0;
			uint32_t j = 0;
			std::tie(i, j) = GetNextCliques(c, messages);
			//	 as long as there a messages from source cluster i to target cluster j
			while (i || j) {
				// inputs = struct('var', [], 'card', [], 'val', []); 
				// add all input messages except from target cluster j
				Factor input{};
				// for k = 1:N
				for (uint32_t k = 0; k < n; ++k) {
					// sum all input messages except from target cluster j
					//	if P.edges(i, k) && k~=j 
					if (c.edges[i][k] && (k != j)) {
						//	inputs = FactoSum(inputs, MESSAGES(k, i));
						input = FactorSum(input, messages[k][i]);
					}// end
				} // end
				// add own initial potential
				//	MESSAGES(i, j) = FactorSum(P.cliqueList(i), inputs);
				messages[i][j] = FactorSum(c.clique_list[i], input);
				// elimiate variables from message from i to j that are not known to target cluster j
				//	eliminateVars = setdiff(P.cliqueList(i).var, P.cliqueList(j).var);
				const auto eliminate_vars = Difference(c.clique_list[i].Var(), c.clique_list[j].Var()).values;
				//	MESSAGES(i, j) = FactorMaxMarginalization(MESSAGES(i, j), eliminateVars);
				messages[i][j] = FactorMaxMarginalization(messages[i][j], eliminate_vars);
				//	[i, j] = GetNextCliques(P, MESSAGES);
				// get next message
				std::tie(i, j) = GetNextCliques(c, messages);
			}// end
		// else
		} else {
			uint32_t i = 0;
			uint32_t j = 0;
			std::tie(i, j) = GetNextCliques(c, messages);
			//	 as long as there a messages from source cluster i to target cluster j
			while (i || j) {
				// inputs = struct('var', [], 'card', [], 'val', []); 
				// multiply all input messages except from target cluster j
				Factor input{};
				// for k = 1:N
				for (uint32_t k = 0; k < n; ++k) {
					// multiply all input messages except from target cluster j
					//	if P.edges(i, k) && k~=j 
					if (c.edges[i][k] && (k != j)) {
						//	inputs = FactorProduct(inputs, MESSAGES(k, i));
						input = FactorProduct(input, messages[k][i]);
					}// end
				} // end
				// multiply with own initial potential
				//	MESSAGES(i, j) = FactorProduct(P.cliqueList(i), inputs);
				messages[i][j] = FactorProduct(c.clique_list[i], input);
				// elimiate variables from message from i to j that are not known to target cluster j
				//	eliminateVars = setdiff(P.cliqueList(i).var, P.cliqueList(j).var);
				const auto eliminate_vars = Difference(c.clique_list[i].Var(), c.clique_list[j].Var()).values;
				//	MESSAGES(i, j) = FactorMarginalization(MESSAGES(i, j), eliminateVars);
				messages[i][j] = FactorMarginalization(messages[i][j], eliminate_vars);
				//	MESSAGES(i, j) = NormalizeFactorValues(MESSAGES(i, j));
				messages[i][j].Normalize();
				//	[i, j] = GetNextCliques(P, MESSAGES);
				// get next message
				std::tie(i, j) = GetNextCliques(c, messages);
			}// end
		}//end
		//
		// Now the clique tree has been calibrated. 
		// Compute the final potentials for the cliques and place them in P.
		//
		//if isMax
		if (is_max) {
			// for all cliques
			//	 for i = 1:N
			for (uint32_t i = 0; i < n; ++i) {
				//	for k = 1:N
				for (uint32_t k = 0; k < n; ++k) {
					//	if P.edges(i, k)  
					if (c.edges[i][k] == 1) {
						// sum initial potential with all incoming messages
						//	P.cliqueList(i) = FactorSum(P.cliqueList(i), MESSAGES(k, i));
						c.clique_list[i] = FactorSum(c.clique_list[i], messages[k][i]);
					}// end
				}// end
			}//	 end    
		} //else
		else {
			// for all cliques
			//	 for i = 1:N
			for (uint32_t i = 0; i < n; ++i) {
				//	for k = 1:N
				for (uint32_t k = 0; k < n; ++k) {
					//	if P.edges(i, k)  
					if (c.edges[i][k] == 1) {
						// mupltiply initial potential with all incoming messages
						//	P.cliqueList(i) = FactorProduct(P.cliqueList(i), MESSAGES(k, i));
						c.clique_list[i] = FactorProduct(c.clique_list[i], messages[k][i]);
					}// end
				}// end
			}//	 end    
		}//end
	} //return

	// COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
	// over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
	// 
	//    M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
	//    evidence E, and a flag isMax, runs exact inference and returns the
	//    final marginals for the variables in the network. If isMax is 1, then
	//    it runs exact MAP inference, otherwise exact inference (sum-prod).
	//    It returns an array of size equal to the number of variables in the 
	//    network where M(i) represents the ith variable and M(i).val represents 
	//    the marginals of the ith variable. 

	// function M = ComputeExactMarginalsBP(F, E, isMax)
	std::vector<Factor> ComputeExactMarginalsBP(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& evidence, bool is_max) {
		//  initialization
		//  you should set it to the correct value in your code
		//M = [];
		//                                                                       
		//  YOUR CODE HERE
		// 
		//  Implement Exact and MAP Inference.
		//                                                                       
		//P = CreateCliqueTree(F, E);
		CliqueTree c = CreateCliqueTree(f, evidence);
		//P = CliqueTreeCalibrate(P, isMax);
		CliqueTreeCalibrate(c, is_max);
		//vars = [];
		std::vector<uint32_t> vars{};
		//for i = 1:length(P.cliqueList)
		for (uint32_t i = 0; i < c.clique_list.size(); ++i) {
			//  vars = union(vars, P.cliqueList(i).var);
			vars = Union(vars, c.clique_list[i].Var()).values;
		}//end
		//
		//M = repmat(struct('var', 0, 'card', 0, 'val', []), length(vars), 1);
		std::vector<Factor> m(vars.size());
		//for i = 1:length(vars)
		for (uint32_t i = 0; i < vars.size(); ++i) {
			//	var = vars(i);   the var for which we compute the marginals
			const auto& var = vars[i];
			//    for j = 1:length(P.cliqueList)
			for (uint32_t j = 0; j < c.clique_list.size(); ++j) {
			// if ismember(var, P.cliqueList(j).var)
				if (!Intersection({ var }, c.clique_list[j].Var()).values.empty()) {
					// if isMax
					if (is_max) {
						// M(var) = FactorMaxMarginalization(P.cliqueList(j), setdiff(P.cliqueList(j).var, [var]));
						m[var] = FactorMaxMarginalization(c.clique_list[j], Difference(c.clique_list[j].Var(), { var }).values);
					} // else
					else {
						// M(var) = FactorMarginalization(P.cliqueList(j), setdiff(P.cliqueList(j).var, [var]));
						m[var] = FactorMarginalization(c.clique_list[j], Difference(c.clique_list[j].Var(), { var }).values);
						// M(var) = NormalizeFactorValues(M(var));
						m[var].Normalize();
					}// end
					// break;
					break;
				} // end
			}// end
		}//end
		return m;
	}//end

	// MAXDECODING Finds the best assignment for each variable from the marginals
	// passed in. Returns A such that A(i) returns the index of the best
	// instantiation for variable i.
	// 
	//    For instance: Let's say we have two variables 1 and 2. 
	//    Marginals for 1 = [0.1, 0.3, 0.6]
	//    Marginals for 2 = [0.92, 0.08]
	//    A(1) = 3, A(2) = 1.
	// 
	//    See also COMPUTEEXACTMARGINALSBP
	//
	//  based on Coursera PGM course by Daphne Koller, Stanford Univerity, 2012
	//
	//function A = MaxDecoding( M )
	std::vector<uint32_t> MaxDecoding(const std::vector<Factor>& m) {
		//
		//  Compute the best assignment for variables in the network.
		//A = zeros(1, length(M));
		std::vector<uint32_t> a(m.size(), 0);
		//for i = 1:length(M)
		for (uint32_t i = 0; i < m.size(); ++i) { // Iterate through variables
			// [maxVal, idx] = max(M(i).val);
			const auto result = std::max_element(m[i].Val().begin(), m[i].Val().end());
			// A(i) = idx;
			a[i] = static_cast<uint32_t>(std::distance(m[i].Val().begin(), result));
		} //end
		return a;//
	}//end


}
	
	
