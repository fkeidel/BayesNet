#include "factor.h"
#include <numeric>
#include <iterator>
#include <iostream>
#include <cassert>

namespace Bayes {
	Factor::Factor(const std::vector<uint32_t>& var, const std::vector<uint32_t>& card, const std::vector<double>& val) :
		var_(var), card_(card), val_(val)
	{
		assert(("card size must be qual to var size", card.size() == var.size()));
	}

	// AssignmentToIndex Convert assignment to index.
	//
	//   I = AssignmentToIndex(A, D) converts an assignment, A, over variables
	//   with cardinality D to an index into the .val vector for a factor. 
	//   If A is a matrix then the function converts each row of A to an index.
	//
	// function I = AssignmentToIndex(A, D)
	//	I = cumprod([1, D(1:end - 1)]) * (A(:) - 1) + 1;
	std::size_t Factor::AssigmentToIndex(const std::vector<uint32_t>& assignment) const
	{
		// card = [1, D(1:end - 1)]
		std::vector<uint32_t> card{ 1 };
		std::copy(card_.begin(), card_.end() - 1, std::back_inserter(card));
		// intervals = cumprod(card)
		std::vector<uint32_t> intervals(card_.size(), 0);
		std::partial_sum(card.begin(), card.end(), intervals.begin(), std::multiplies<uint32_t>());
		//	I = cumprod([1, D(1:end - 1)]) * (A(:) - 1) + 1;
		return std::inner_product(intervals.begin(), intervals.end(), assignment.begin(), 0);
	}

	// IndexToAssignment Convert index to variable assignment.
	//
	//   A = IndexToAssignment(I, D) converts an index, I, into the .val vector
	//   into an assignment over variables with cardinality D. If I is a vector, 
	//   then the function produces a matrix of assignments, one assignment 
	//   per row.
	//
	//   See also AssignmentToIndex.m and FactorTutorial.m
	//
	//function A = IndexToAssignment(I, D)
	//	A = mod(floor(repmat(I(:) - 1, 1, length(D)) . / repmat(cumprod([1, D(1:end - 1)]), length(I), 1)), ...
	//		repmat(D, length(I), 1)) + 1;
	std::vector<uint32_t> Factor::IndexToAssignment(size_t index) const
	{
		std::vector<uint32_t> assignment(card_.size(), 0);
		std::vector<uint32_t> intervals(card_.size(), 0);
		// card = [1, D(1:end - 1)]
		std::vector<uint32_t> card{ 1 };
		std::copy(card_.begin(), card_.end() - 1, std::back_inserter(card));
		// intervals = cumprod(card)
		std::partial_sum(card.begin(), card.end(), intervals.begin(), std::multiplies<uint32_t>());
		//	A = mod(floor(repmat(I(:) - 1, 1, length(D)) . / repmat(cumprod([1, D(1:end - 1)]), length(I), 1)), ...
		//		repmat(D, length(I), 1)) + 1;
		for (size_t i = 0; i < card_.size(); ++i) {
			assignment[i] = (index / intervals[i]) % card_[i];
		}
		return assignment;
	}

	double Factor::GetValueOfAssignment(const std::vector<uint32_t>& assignment)
	{
		return val_[AssigmentToIndex(assignment)];
	}

	void Factor::SetValueOfAssignment(const std::vector<uint32_t>& assignment, double value)
	{
		val_[AssigmentToIndex(assignment)] = value;
	}

	void Factor::SetVar(const std::vector<uint32_t>& var)
	{
		var_ = var;
	}

	void Factor::SetCard(const std::vector<uint32_t>& card)
	{
		card_ = card;
	}

	void Factor::SetCard(size_t index, uint32_t card)
	{
		card_[index] = card;
	}

	void Factor::SetVal(const std::vector<double>& val)
	{
		val_ = val;
	}

	void Factor::SetVal(size_t index, double val)
	{
		val_[index] = val;
	}

	// function [CPD] = CPDFromFactor(F, Y)
	//  Reorder the var, card and val fields of Fnew so that the last var is the 
	//  child variable.
	// original Matlab code: Copyright (C) Daphne Koller, Stanford University, 2012
	Factor Factor::CPD(uint32_t y) {
		//  YIndexInF = find(F.var == Y);
		const auto& it = std::find(var_.begin(), var_.end(), y);
		assert(("y must be in var_", it != var_.end())); 
		const auto y_index_in_f = std::distance(var_.begin(), it);

		//  this.card = F.card( YIndexInF );
		const auto y_card = card_[y_index_in_f];

		// Parents is a dummy factor
		//  Parents.var = F.var(find(F.var ~= Y));
		//  Parents.card = F.card(find(F.var ~= Y));
		//  Parents.val = ones(prod(Parents.card),1);
		std::vector<uint32_t> parents_var;
		std::vector<uint32_t> parents_card;
		for (size_t i = 0; i < var_.size(); ++i) {
			if (var_[i] != y) {
				parents_var.push_back(var_[i]);
				parents_card.push_back(card_[i]);
			}
		}
		std::vector<double> parents_val(std::accumulate(parents_card.begin(), parents_card.end(), 1, std::multiplies<uint32_t>()), 1.0);
		Factor parents{parents_var, parents_card, parents_val};

		//  Fnew.var = [Parents.var Y];
		//  Fnew.card = [Parents.card this.card];
		std::vector<uint32_t> fnew_var(parents_var);
		fnew_var.push_back(y);
		std::vector<uint32_t> fnew_card(parents_card);
		fnew_card.push_back(y_card);

		Factor fnew{fnew_var, fnew_card, std::vector<double>(val_.size(),0.0)};

		//  for i=1:length(F.val)
		for (size_t i = 0; i < val_.size(); ++i) {
			//    A = IndexToAssignment(i, F.card);
			std::vector<uint32_t> a = IndexToAssignment(i);
			//    y = A(YIndexInF);
			const auto a_y = a[y_index_in_f];
			//    A( YIndexInF ) = [];
			a.erase(a.begin() + y_index_in_f);
			//    A = [A y];
			a.push_back(a_y);
			//    j = AssignmentToIndex(A, Fnew.card);
			const auto j = fnew.AssigmentToIndex(a);
			//    Fnew.val(j) = F.val(i);
			fnew.SetVal(j, val_[i]);
		} //  end
		
		// For each assignment of Parents...
		// for i=1:length(Parents.val)
		for (size_t i = 0; i < parents_val.size(); ++i) {
			// A = IndexToAssignment(i, Parents.card);
			const auto a = parents.IndexToAssignment(i);
			// SumValuesForA = 0;
			double sum_values_for_a{ 0.0 };
			// for j=1:this.card
			for (size_t j = 0; j < y_card; ++j) {
				// A_augmented = [A j];
				auto a_augmented(a);
				a_augmented.push_back(j);
				// idx = AssignmentToIndex(A_augmented, Fnew.card);
				const auto idx = fnew.AssigmentToIndex(a_augmented);
				// SumValuesForA = SumValuesForA + Fnew.val( idx );
				sum_values_for_a += fnew.Val()[idx];
			} // end  
			// for j=1:this.card
			for (size_t j = 0; j < y_card; ++j) {
				// A_augmented = [A j];
				auto a_augmented(a);
				a_augmented.push_back(j);
				// idx = AssignmentToIndex(A_augmented, Fnew.card);
				const auto idx = fnew.AssigmentToIndex(a_augmented);
				// Fnew.val( idx ) = Fnew.val( idx )  / SumValuesForA;
				fnew.SetVal(idx, fnew.Val()[idx] / sum_values_for_a);
			} //    end  
		} //  end
		//  CPD = Fnew;
		return fnew;
	}

	// FactorProduct Computes the product of two factors.
	//   c = FactorProduct(a,b) computes the product between two factors, a and b,
	//   where each factor is defined over a set of variables with given dimension.
	//   The factor data structure has the following fields:
	//       .var    Vector of variables in the factor, e.g. [1 2 3]
	//       .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
	//       .val    Value table of size prod(.card)
	//
	//   See also FactorMarginalization, Factor::IndexToAssignment, and
	//   Factor::AssignmentToIndex
	//
	// original Matlab sources: Copyright (C) Daphne Koller, Stanford University, 2012

	Factor FactorProduct(const Factor& a, const Factor& b) {
		// Check for empty factors
		if (a.IsEmpty()) return b;
		if (b.IsEmpty()) return a;

		// Check that variables in both A and B have the same cardinality
		SetOperationResult<uint32_t> intersection = Intersection(a.Var(), b.Var());
		if (!intersection.values.empty()) {
			// A and B have at least 1 variable in common
			const auto& iA = intersection.left_indices;
			const auto& iB = intersection.right_indices;
			// assert(all(A.card(iA) == B.card(iB)))
			for (size_t i = 0; i < intersection.values.size(); ++i)
			{
				assert(("Dimensionality mismatch in factors", a.Card()[iA[i]] == b.Card()[iB[i]]));
			}
		}

		// Set the variables of c
		SetOperationResult<uint32_t> union_result = Union(a.Var(), b.Var());
		const auto& c_var = union_result.values;
		const auto& mapA = union_result.left_indices;
		const auto& mapB = union_result.right_indices;

		// Set the cardinality of variables in c
		std::vector<uint32_t> c_card(c_var.size(), 0);  // C.card = zeros(1, length(C.var));
		// C.card(mapA) = A.card;
		for (size_t i = 0; i < mapA.size(); ++i) {
			c_card[mapA[i]] = a.Card()[i];
		}
		// C.card(mapB) = B.card;
		for (size_t i = 0; i < mapB.size(); ++i) {
			c_card[mapB[i]] = b.Card()[i];
		}

		Factor c{ union_result.values,c_card, {} };

		// create assignments of c
		// assignments = IndexToAssignment(1:prod(C.card), C.card);
		std::vector<std::vector<uint32_t>> assignments;
		// assignments_size = prod(C.card)
		const auto assignments_size = std::accumulate(c_card.begin(), c_card.end(), 1, std::multiplies<uint32_t>());
		for (size_t i = 0; i < assignments_size; ++i) {
			assignments.push_back(c.IndexToAssignment(i));
		}

		// indxA = AssignmentToIndex(assignments(:, mapA), A.card);
		std::vector<size_t> indxA;
		for (size_t i = 0; i < assignments_size; ++i) {
			std::vector<uint32_t> assignment;
			for (size_t j = 0; j < mapA.size(); ++j) {
				assignment.push_back(assignments[i][mapA[j]]);
			}
			indxA.push_back(a.AssigmentToIndex(assignment));
		}

		// indxB = AssignmentToIndex(assignments(:, mapB), B.card);
		std::vector<size_t> indxB;
		for (size_t i = 0; i < assignments_size; ++i) {
			std::vector<uint32_t> assignment;
			for (size_t j = 0; j < mapB.size(); ++j) {
				assignment.push_back(assignments[i][mapB[j]]);
			}
			indxB.push_back(b.AssigmentToIndex(assignment));
		}

		// populate the factor values of c
		// C.val = A.val(indxA).*B.val(indxB);
		std::vector<double> c_values;
		for (size_t i = 0; i < assignments_size; ++i) {
			c_values.push_back(a.Val()[indxA[i]] * b.Val()[indxB[i]]);
		}

		c.SetVal(c_values);

		return c;
	}

	// FactorMarginalization Sums given variables out of a factor.
	//   b = FactorMarginalization(a,v) computes the factor with the variables
	//   in v summed out. The factor data structure has the following fields:
	//       .var    Vector of variables in the factor, e.g. [1 2 3]
	//       .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
	//       .val    Value table of size prod(.card)
	//
	//   The resultant factor should have at least one variable remaining or this
	//   function will throw an error.
	// 
	//   See also FactorProduct, Factor::IndexToAssignment, and Factor::AssignmentToIndex
	//
	// original Matlab code: Copyright (C) Daphne Koller, Stanford University, 2012

	Factor FactorMarginalization(const Factor& a, const std::vector<uint32_t>& v)
	{
		// Check for empty factor or variable list
		if (a.Var().empty() || v.empty()) return a;

		// Construct the output factor over A.var \ v (the variables in A.var that are not in v)
		SetOperationResult<uint32_t> diff = Difference(a.Var(), v);
		const auto& b_var = diff.values;
		const auto& mapB = diff.left_indices;

		// Check for empty resultant factor
		if (b_var.empty()) return { {},{},{std::accumulate(a.Val().begin(), a.Val().end(),0.0) } };

		// Initialize B.card and B.val

		std::vector<uint32_t> b_card(b_var.size(), 0);
		// B.card = A.card(mapB);
		for (size_t i = 0; i < mapB.size(); ++i) {
			b_card[i] = a.Card()[mapB[i]];
		}

		// B.val = zeros(1, prod(B.card));
		// b_val.size = prod(B.card)
		const auto b_val_size = std::accumulate(b_card.begin(), b_card.end(), 1, std::multiplies<uint32_t>());
		std::vector<double> b_val(b_val_size, 0);

		// assignments = IndexToAssignment(1:length(A.val), A.card);
		std::vector<std::vector<uint32_t>> assignments;
		for (size_t i = 0; i < a.Val().size(); ++i) {
			assignments.push_back(a.IndexToAssignment(i));
		}

		Factor b{ b_var, b_card, {} };

		// indxB = AssignmentToIndex(assignments(:, mapB), B.card);
		std::vector<size_t> indxB;
		for (size_t i = 0; i < a.Val().size(); ++i) {
			std::vector<uint32_t> assignment;
			for (size_t j = 0; j < mapB.size(); ++j) {
				assignment.push_back(assignments[i][mapB[j]]);
			}
			indxB.push_back(b.AssigmentToIndex(assignment));
		}

		// Correctly populate the factor values of B
		// for i = 1:length(A.val),
		//	B.val(indxB(i)) = B.val(indxB(i)) + A.val(i);
		// end;
		for (size_t i = 0; i < a.Val().size(); ++i) {
			b_val[indxB[i]] = b_val[indxB[i]] + a.Val()[i];
		}

		b.SetVal(b_val);

		return b;
	}

//ObserveEvidence Modify a vector of factors given some evidence.
//  F = ObserveEvidence(F, E) sets all entries in the vector of factors, F,
//  that are not consistent with the evidence, E, to zero. F is a vector of
//  factors, each a data structure with the following fields:
//    .var    Vector of variables in the factor, e.g. [1 2 3]
//    .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
//    .val    Value table of size prod(.card)
//  E is an N-by-2 matrix, where each row consists of a variable/value pair. 
//    Variables are in the first column and values are in the second column.
//  NOTE - DOES NOT RENORMALIZE THE FACTOR VALUES 
//
// original Matlab code: Copyright (C) Daphne Koller, Stanford University, 2012
//
//function F = ObserveEvidence(F, E, normalize)
	void ObserveEvidence(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& e)
	{
		//  Iterate through all evidence
		//  for i = 1:size(E, 1),
		//    v = E(i, 1); variable
		//    x = E(i, 2); value
		// 
		for (const auto& evidence : e) {
			uint32_t v = evidence.first;
			uint32_t x = evidence.second;

			//    Iterate through the factors
			//    for j = 1:length(F),
			for (size_t j = 0; j < f.size(); ++j) 
			{
				// Does factor contain variable?
				// indx = find(F(j).var == v);
				const auto& it = std::find(f[j].Var().begin(), f[j].Var().end(), v);
				// if (~isempty(indx)),
				if (it != f[j].Var().end()) {
					const auto indx = std::distance(f[j].Var().begin(), it);

					//	Check validity of evidence
					//	if (x > F(j).card(indx) || x < 0 ),
					//	  error(['Invalid evidence, X_', int2str(v), ' = ', int2str(x)]);
					//	end;
					assert(("Invalid evidence", x < f[j].Card()[indx]));

					//	Adjust the factor F(j) to account for observed evidence
					//	For each value (1-1 map between assignment and values)
					//	for k = 1:length(F(j).val),
					for (size_t k = 0; k < f[j].Val().size(); ++k) {
						// get assignment for this index
						// A = IndexToAssignment(k, F(j).card);
						const auto a = f[j].IndexToAssignment(k);
						//
						//	  indx = index of evidence variable in this factor
						//	  if (A(indx) ~= x),
						//	    F(j).val(k) = 0;
						//	  end;
						if (a[indx] != x) {
							f[j].SetVal(k, 0.0);
						}
					} // end for k = 1:length(F(j).val)

					//	Check validity of evidence / resulting factor
					//	if (all(F(j).val == 0)),
					//	  warning(['Factor ', int2str(j), ' makes variable assignment impossible']);
					//	end;
					if (std::all_of(f[j].Val().begin(), f[j].Val().end(), [](double d) {return d == 0.0; }))
						std::cout << "Factor " << j << "makes variable assignment impossible" << std::endl;
				} //end if (!isempty(index))		
			} //   end for j = 1:length(F),
		} //  end for i = 1:size(E, 1),
	}

	// EliminateVar
	// Function used in production of clique trees
	//	F = list of factors
	//	E = adjacency matrix for variables
	//	Z = variable to eliminate

	//	original Matlab code: Copyright(C) Daphne Koller, Stanford University, 2012

	//	function[newF E] = EliminateVar(F, E, Z)
	void EliminateVar(std::vector<Factor>& f, std::vector<std::vector<uint32_t>>& e, uint32_t z)
	{
		//	Index of factors to multiply(b / c they contain Z)
		//	useFactors = [];
		std::vector<size_t> useFactors;

		//Union of scopes of factors to multiply
		//	scope = [];
		std::vector<uint32_t> scope;

		// for i = 1:length(F)
		for (size_t i = 0; i < f.size(); ++i) {
			//if any(F(i).var == Z)
			if (std::any_of(f[i].Var().begin(), f[i].Var().end(), [z](uint32_t i) {return i == z; })) {
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
		//	nonUseFactors = setdiff(1:length(F), [useFactors]);
		std::vector<size_t> range_f(f.size(), 0);
		std::iota(range_f.begin(), range_f.end(), 0);
		std::vector<size_t> nonUseFactors = Difference(range_f, useFactors).values;

		//	newF = list of factors we will return
		std::vector<Factor> newF(nonUseFactors.size());
		//for i = 1:length(nonUseFactors)
		for (size_t i = 0; i < nonUseFactors.size(); ++i) {
			//	newF(i) = F(nonUseFactors(i));
			newF[i] = f[nonUseFactors[i]];
			// newmap = ?
			//	newmap(nonUseFactors(i)) = i;
		} //end

		//	Multiply factors which involve Z->newFactor
		//	newFactor = struct('var', [], 'card', [], 'val', []);
		Factor newFactor{};
		//for i = 1:length(useFactors)
		for (size_t i = 0; i < useFactors.size(); ++i) {
			//	newFactor = FactorProduct(newFactor, F(useFactors(i)));
			newFactor = FactorProduct(newFactor, f[useFactors[i]]);
		} //end

		//	newFactor = FactorMarginalization(newFactor, Z);
		newFactor = FactorMarginalization(newFactor, { z });
		//newF(length(nonUseFactors) + 1) = newFactor;
		newF.push_back(newFactor);
		f = newF;
	}

	//	VariableElimination takes in a list of factors F and a list of variables to eliminate
	//	and returns the resulting factor after running sum-product to eliminate
	//	the given variables.
	//	
	//	Fnew = VariableElimination(F, Z)
	//	F = list of factors
	//	Z = list of variables to eliminate

	//	original Matlab code: Copyright(C) Daphne Koller, Stanford University, 2012
	//	function Fnew = VariableElimination(F, Z)
	std::vector<Factor> VariableElimination(std::vector<Factor>& f, const std::vector<uint32_t>& z)
	{
		//	List of all variables
		//	V = unique([F(:).var]);
		const auto v{ UniqueVars(f) };

		// Setting up the adjacency matrix.
		auto edges{ SetUpAdjacencyMatrix(v,f) };

		//	variablesConsidered = 0;
		size_t variablesConsidered{ 0 };
		//while variablesConsidered < length(Z)
		while (variablesConsidered < z.size())
		{
			//	Using Min - Neighbors where you prefer to eliminate the variable that has
			//	the smallest number of edges connected to it.
			//	Everytime you enter the loop, you look at the state of the graph and
			//	pick the variable to be eliminated.
			//	bestVariable = 0;
			uint32_t bestVariable{ 0 };
			//	bestScore = inf;
			uint32_t bestScore{ std::numeric_limits<uint32_t>::max() };
			//	for i = 1:length(Z)
			for (size_t i = 0; i < z.size(); ++i) {
				//  idx = Z(i);
				const auto idx = z[i];
				//	score = sum(edges(idx, :));
				const auto score = std::accumulate(edges[idx].begin(), edges[idx].end(), 0);
				//	if score > 0 && score < bestScore
				if ((score > 0) && (score < bestScore)) {
					//	bestScore = score;
					bestScore = score;
					//	bestVariable = idx;
					bestVariable = idx;
				} //  end
			} // end
			//	variablesConsidered = variablesConsidered + 1;
			++variablesConsidered;
			//[F, edges] = EliminateVar(F, edges, bestVariable);
			EliminateVar(f, edges, bestVariable);
		} //end
		return f;
	}

	// ComputeJointDistribution Computes the joint distribution defined by a set
	// of given factors
	// 
	//   Joint = ComputeJointDistribution(F) computes the joint distribution
	//   defined by a set of given factors
	// 
	//   Joint is a factor that encapsulates the joint distribution given by F
	//   F is a vector of factors (struct array) containing the factors 
	//     defining the distribution
	// 
	Factor ComputeJointDistribution(const std::vector<Factor>& f) {
		Factor joint{};
		for (size_t i = 0; i < f.size(); ++i) {
			//Joint = FactorProduct(Joint, F(i));
			joint = FactorProduct(joint, f[i]);
		} //end
		return joint;
	}

	//ComputeMarginal Computes the marginal over a set of given variables
	//   M = ComputeMarginal(V, F, E) computes the marginal over variables V
	//   in the distribution induced by the set of factors F, given evidence E
	//
	//   M is a factor containing the marginal over variables V
	//   V is a vector containing the variables in the marginal e.g. [1 2 3] for
	//     X_1, X_2 and X_3.
	//   F is a vector of factors (struct array) containing the factors 
	//     defining the distribution
	//   E is an N-by-2 matrix, each row being a variable/value pair. 
	//     Variables are in the first column and values are in the second column.
	//     If there is no evidence, pass in the empty matrix [] for E.
	Factor ComputeMarginal(const std::vector<uint32_t>& v, std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& e) {
		// Check for empty factor list
		if (f.empty()) return {};
		ObserveEvidence(f, e);
		Factor joint = ComputeJointDistribution(f);
		Factor m = FactorMarginalization(joint, Difference(joint.Var(), v).values);
		// M.val = M.val. / sum(M.val);
		const auto sum = std::accumulate(m.Val().begin(), m.Val().end(), 0.0);
		for (size_t i = 0; i < m.Val().size(); ++i) {
			m.SetVal(i, m.Val()[i] / sum);
		}
		return m;
	}

	std::vector<uint32_t> UniqueVars(std::vector<Factor> f) {
		//	V = unique([F(:).var]);
		std::vector<uint32_t> v;
		for (const auto& factor : f)
		{
			v.insert(v.end(), factor.Var().begin(), factor.Var().end());
		}
		std::sort(v.begin(), v.end());
		const auto last = std::unique(v.begin(), v.end());
		v.erase(last, v.end());
		return v;
	}

	// Setting up the adjacency matrix
	// @param v list of all variables in the adjacency matrix
	// @param f list of factors containing the variables
	std::vector<std::vector<uint32_t>> SetUpAdjacencyMatrix(const std::vector<uint32_t>& v, const std::vector<Factor>& f) {
		//	edges = zeros(length(V));
		std::vector<std::vector<uint32_t>> edges(v.size(), std::vector<uint32_t>(v.size(), 0));
		//for i = 1:length(F)
		for (size_t i = 0; i < f.size(); ++i) {
			//	for j = 1 : length(F(i).var)
			for (size_t j = 0; j < f[i].Var().size(); ++j) {
				// for k = 1 : length(F(i).var)
				for (size_t k = 0; k < f[i].Var().size(); ++k) {
					// edges(F(i).var(j), F(i).var(k)) = 1;
					edges[f[i].Var()[j]][f[i].Var()[k]] = 1;
				} // end
			} // end
		} //end
		return edges;
	}

	std::ostream& operator<<(std::ostream& out, const Factor& v) {
		out << "var=" << v.Var() << "card=" << v.Card() << "val=" << v.Val() << std::endl;
		return out;
	}
}
