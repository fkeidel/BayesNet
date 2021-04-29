// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

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
	//   AssignmentToIndex(assignment) converts an assignment over variables
	//   to an index into the .val vector
	//   
	std::size_t Factor::AssigmentToIndex(const std::vector<uint32_t>& assignment) const
	{
		std::vector<uint32_t> card{ 1 };
		std::copy(card_.begin(), card_.end() - 1, std::back_inserter(card));
		std::vector<uint32_t> intervals(card_.size(), 0);
		std::partial_sum(card.begin(), card.end(), intervals.begin(), std::multiplies<uint32_t>());
		return std::inner_product(intervals.begin(), intervals.end(), assignment.begin(), 0);
	}

	// IndexToAssignment Convert index to variable assignment.
	//
	//   IndexToAssignment(index) converts an index into the .val vector
	//   into an assignment over variables 
	//
	std::vector<uint32_t> Factor::IndexToAssignment(size_t index) const
	{
		std::vector<uint32_t> assignment(card_.size(), 0);
		std::vector<uint32_t> intervals(card_.size(), 0);
		std::vector<uint32_t> card{ 1 };
		std::copy(card_.begin(), card_.end() - 1, std::back_inserter(card));
		std::partial_sum(card.begin(), card.end(), intervals.begin(), std::multiplies<uint32_t>());
		for (size_t i = 0; i < card_.size(); ++i) {
			assignment[i] = (index / intervals[i]) % card_[i];
		}
		return assignment;
	}

	double Factor::GetValueOfAssignment(const std::vector<uint32_t>& assignment) const
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

	// CPD(y)  create CPD from factor
	//   Reorder the var, card and val fields of the factor so that the last var is the 
	//   child variable.
	Factor Factor::CPD(uint32_t y) 
	{
		const auto& it = std::find(var_.begin(), var_.end(), y);
		assert(("y must be in var_", it != var_.end())); 
		const auto y_index_in_f = std::distance(var_.begin(), it);

		const auto y_card = card_[y_index_in_f];

		// temp factor for parents used for assignment/index operations
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

		// the new factor starts with the parents
		std::vector<uint32_t> cpd_var(parents_var);
		std::vector<uint32_t> cpd_card(parents_card);
		// append child as last element
		cpd_var.push_back(y);
		cpd_card.push_back(y_card);
		// create new factor
		Factor cpd{cpd_var, cpd_card, std::vector<double>(val_.size(), 0.0)};

		// for all values
		for (size_t i = 0; i < val_.size(); ++i) 
		{
			// delete assignment to var y from original assignment and append it as last element
			std::vector<uint32_t> a = IndexToAssignment(i);
			const auto a_y = a[y_index_in_f];
			a.erase(a.begin() + y_index_in_f);
			a.push_back(a_y);
			// copy value from current factor and old index i into new factor at the new index j
			const auto j = cpd.AssigmentToIndex(a);
			cpd.SetVal(j, val_[i]);
		} //  end
		
		// normalize: for each joint assignment to parents, normalize the values to sum to 1
		for (size_t i = 0; i < parents_val.size(); ++i) 
		{
			const auto a = parents.IndexToAssignment(i);
			// sum up values of child y for this parent assignment a
			double sum_values_for_a{ 0.0 };
			for (uint32_t j = 0; j < y_card; ++j) 
			{
				auto a_augmented(a);
				a_augmented.push_back(j);
				const auto idx = cpd.AssigmentToIndex(a_augmented);
				sum_values_for_a += cpd.Val(idx);
			} 
			// normalize: divide all values of child y for this parent assignment by the sum
			for (uint32_t j = 0; j < y_card; ++j) 
			{
				auto a_augmented(a);
				a_augmented.push_back(j);
				const auto idx = cpd.AssigmentToIndex(a_augmented);
				auto normalized_val{ 0.0 };
				if (sum_values_for_a != 0.0)
				{
					normalized_val = cpd.Val(idx) / sum_values_for_a;
				}
				cpd.SetVal(idx, normalized_val);
			}
		}
		return cpd;
	}

	// Normalize normalizes the values to sum to 1.
	void Factor::Normalize() 
	{
		const auto sum = std::accumulate(val_.begin(), val_.end(), 0.0);
		std::for_each(val_.begin(), val_.end(), [sum](auto& val) { val /= sum; });
	}

	// Marginalize Sums given variables out of a factor.
	//   Marginalize(z) computes the factor with the variables
	//   in z summed out. 
	//
	Factor Factor::Marginalize(const std::vector<uint32_t>& z) const 
	{
		// Check for empty factor or variable list
		if (var_.empty() || z.empty()) return *this;

		// Construct the output factor over var \ v (the variables in var that are not in v)
		SetOperationResult<uint32_t> diff = Difference(var_, z);
		const auto& new_var = diff.values;
		const auto& map_new_var = diff.left_indices;

		// Check for empty resultant factor
		if (new_var.empty()) {
			return { {}, {}, { std::accumulate(val_.begin(), val_.end(), 0.0) } };
		}

		// initialize new card
		std::vector<uint32_t> new_card(new_var.size(), 0);
		for (size_t i = 0; i < map_new_var.size(); ++i) {
			new_card[i] = card_[map_new_var[i]];
		}

		// initialize new val array
		const auto f_new_val_size = std::accumulate(new_card.begin(), new_card.end(), 1, std::multiplies<uint32_t>());
		std::vector<double> new_val(f_new_val_size, 0.0);

		// list of all assignments for original factor
		std::vector<std::vector<uint32_t>> assignments;
		for (size_t i = 0; i < val_.size(); ++i) {
			assignments.push_back(IndexToAssignment(i));
		}

		Factor f_new{ new_var, new_card, {} };

		std::vector<size_t> new_index(val_.size());
		for (size_t i = 0; i < val_.size(); ++i) {
			std::vector<uint32_t> new_assignment;
			for (size_t j = 0; j < map_new_var.size(); ++j) {
				new_assignment.push_back(assignments[i][map_new_var[j]]);
			}
			new_index[i] = f_new.AssigmentToIndex(new_assignment);
		}

		// sum up values that have the same new index
		for (size_t i = 0; i < val_.size(); ++i) {
			new_val[new_index[i]] = new_val[new_index[i]] + val_[i];
		}

		f_new.SetVal(new_val);

		return f_new;
	}

	// FactorMaxMarginalization Takes the max of given variables when marginalizing out of a factor.
	//   FactorMaxMarginalization(z) takes in a factor and a set of variables to
	//   marginalize out. For each assignment to the remaining variables, it finds the maximum
	//   factor value over all possible assignments to the marginalized variables.
	//   The resultant factor should have at least one variable remaining.
	//
	Factor Factor::MaxMarginalize(const std::vector<uint32_t>& z) const
	{
		// Check for empty factor or variable list
		if (var_.empty() || z.empty()) return *this;

		// Construct the output factor over A.var \ v (the variables in A.var that are not in v)
		SetOperationResult<uint32_t> diff = Difference(var_, z);
		const auto& new_var = diff.values;
		const auto& map_new_var = diff.left_indices;

		// check for empty resultant factor
		assert(("resultant factor is empty", !new_var.empty()));

		// initialize new card
		std::vector<uint32_t> new_card(new_var.size(), 0);
		for (size_t i = 0; i < map_new_var.size(); ++i) {
			new_card[i] = card_[map_new_var[i]];
		}

		// initialize new val array
		const auto new_val_size = std::accumulate(new_card.begin(), new_card.end(), 1, std::multiplies<uint32_t>());
		std::vector<double> new_val(new_val_size, std::numeric_limits<uint32_t>::min());

		// list of all assignments for original factor
		std::vector<std::vector<uint32_t>> assignments;
		for (size_t i = 0; i < val_.size(); ++i) {
			assignments.push_back(IndexToAssignment(i));
		}

		Factor f_new{ new_var, new_card, {} };

		std::vector<size_t> new_index;
		for (size_t i = 0; i < val_.size(); ++i) {
			std::vector<uint32_t> new_assignment;
			for (size_t j = 0; j < map_new_var.size(); ++j) {
				new_assignment.push_back(assignments[i][map_new_var[j]]);
			}
			new_index.push_back(f_new.AssigmentToIndex(new_assignment));
		}

		// select max value with the same new index
		for (size_t i = 0; i < val_.size(); ++i) {
			if (new_val[new_index[i]] == std::numeric_limits<uint32_t>::min()) {
				// new  val not initialized yet
				new_val[new_index[i]] = val_[i];
			}
			else {
				new_val[new_index[i]] = std::max(new_val[new_index[i]], val_[i]);
			}
		}

		f_new.SetVal(new_val);

		return f_new;
	}

	// ObserveEvidence Modify the factors given some evidence.
	//  F = ObserveEvidence(F, E) sets all entries in the factor,
	//  that are not consistent with the evidence, E, to zero. 
	//  E is an N-by-2 matrix, where each row consists of a variable/value pair. 
	//    Variables are in the first column and values are in the second column.
	//  NOTE - DOES NOT RENORMALIZE THE FACTOR VALUES 
	//
	// function F = ObserveEvidence(F, E, normalize)
	void Factor::ObserveEvidence(const Evidence& e)
	{
		//  Iterate through all evidence
		//  for i = 1:size(E, 1),
		//    v = E(i, 1); variable
		//    x = E(i, 2); value
		for (const auto& evidence : e) {
			uint32_t v = evidence.first;
			uint32_t x = evidence.second;

			// Does factor contain variable?
			// indx = find(F(j).var == v);
			const auto& it = std::find(var_.begin(), var_.end(), v);
			// if (~isempty(indx)),
			if (it != var_.end()) {
				const auto indx = std::distance(var_.begin(), it);

				//	Check validity of evidence
				//	if (x > F(j).card(indx) || x < 0 ),
				//	  error(['Invalid evidence, X_', int2str(v), ' = ', int2str(x)]);
				//	end;
				assert(("Invalid evidence", x < card_[indx]));

				//	Adjust the factor F(j) to account for observed evidence
				//	For each value (1-1 map between assignment and values)
				//	for k = 1:length(F(j).val),
				for (size_t k = 0; k < val_.size(); ++k) {
					// get assignment for this index
					// A = IndexToAssignment(k, F(j).card);
					const auto a = IndexToAssignment(k);
					//
					//	  indx = index of evidence variable in this factor
					//	  if (A(indx) ~= x),
					//	    F(j).val(k) = 0;
					//	  end;
					if (a[indx] != x) {
						SetVal(k, 0.0);
					}
				} // end for k = 1:length(F(j).val)

				//	Check validity of evidence / resulting factor
				//	if (all(F(j).val == 0)),
				//	  warning(['Factor ', int2str(j), ' makes variable assignment impossible']);
				//	end;
				if (std::all_of(val_.begin(), val_.end(), [](double d) {return d == 0.0; }))
					std::cout << "Warning: variable new_assignment impossible" << std::endl;
			} //end if (!isempty(index))		
		} //  end for i = 1:size(E, 1),
	}


	Factor FactorArithmetic(const Factor& a, const Factor& b, const FactorValueOp& op)
	{
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
				assert(("Dimensionality mismatch in factors", a.Card(iA[i]) == b.Card(iB[i])));
			}
		}

		// Set the variables of c and construct the mapping between variables in A and B and variables in C.
		// In the code below, we have that
		//	
		//	mapA(i) = j, if and only if, A.var(i) == C.var(j)
		//	
		// and similarly
		//	
		// map_var_new(i) = j, if and only if, B.var(i) == C.var(j)
		//	
		// For example, if A.var = [3 1 4], B.var = [4 5], and C.var = [1 3 4 5],
		//	then, mapA = [2 1 3] and map_var_new = [3 4]; mapA(1) = 2 because A.var(1) = 3
		//	and C.var(2) = 3, so A.var(1) == C.var(2).

		SetOperationResult<uint32_t> union_result = Union(a.Var(), b.Var());
		const auto& c_var = union_result.values;
		const auto& mapA = union_result.left_indices;
		const auto& mapB = union_result.right_indices;

		// Set the cardinality of variables in c
		std::vector<uint32_t> c_card(c_var.size(), 0);  // C.card = zeros(1, length(C.var));
		// C.card(mapA) = A.card;
		for (size_t i = 0; i < mapA.size(); ++i) {
			c_card[mapA[i]] = a.Card(i);
		}
		// C.card(map_var_new) = B.card;
		for (size_t i = 0; i < mapB.size(); ++i) {
			c_card[mapB[i]] = b.Card(i);
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

		// indxB = AssignmentToIndex(assignments(:, map_var_new), B.card);
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
			c_values.push_back(op(a.Val(indxA[i]), b.Val(indxB[i])));
		}

		c.SetVal(c_values);

		return c;
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

	Factor FactorProduct(const Factor& a, const Factor& b) {
		return FactorArithmetic(a, b, FactorValueMultiply{});
	}

	//	 FactorSum Computes the sum of two factors.
	//   C = FactorSum(A,B) computes the sum of two factors, A and B,
	//   where each factor is defined over a set of variables with given dimension.
	//   The factor data structure has the following fields:
	//       .var    Vector of variables in the factor, e.g. [1 2 3]
	//       .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
	//       .val    Value table of size prod(.card) -- values should be the
	//               logs of the true values
	//
	//   See also FactorMaxMarginalization.m, IndexToAssignment.m, and
	//   AssignmentToIndex.m
	//
	// based on Coursera PGM course by Daphne Koller, Stanford Univerity, 2012
	//
	// function C = FactorSum(A, B)
	Factor FactorSum(const Factor& a, const Factor& b) {
		return FactorArithmetic(a, b, FactorValueAdd{});
	} //end


// ObserveEvidence Modify a vector of factors given some evidence.
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
// function F = ObserveEvidence(F, E, normalize)
	void ObserveEvidence(std::vector<Factor>& f, const Evidence& e)
	{
		//    Iterate through the factors
		//    for j = 1:length(F),
		for (auto& factor : f)
		{
			factor.ObserveEvidence(e);
		} //  end for j = 1:length(F),
	}

	// EliminateVar
	// Function used in production of clique trees
	//	F = list of factors
	//	E = adjacency matrix for variables
	//	Z = variable to eliminate

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
		newFactor = newFactor.Marginalize({ z });
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

	//	function Fnew = VariableElimination(F, Z)
	void VariableElimination(std::vector<Factor>& f, const std::vector<uint32_t>& z)
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
				const auto score = std::accumulate(edges[idx].begin(), edges[idx].end(), 0U);
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

	// SimpleComputeMarginal 
	// Computes the marginal over a set of given variables by creating one large factor and then summing out
	//   M = SimpleComputeMarginal(V, F, E) computes the marginal over variables V
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
	Factor SimpleComputeMarginal(const std::vector<uint32_t>& v, std::vector<Factor>& f, const Evidence& e) {
		// Check for empty factor list
		if (f.empty()) return {};
		ObserveEvidence(f, e);
		Factor joint = ComputeJointDistribution(f);
		Factor m = joint.Marginalize(Difference(joint.Var(), v).values);
		// M.val = M.val. / sum(M.val);
		m.Normalize();
		return m;
	}

	Factor VariableEliminationComputeExactMarginalBP(const uint32_t v, std::vector<Factor>& f, const Evidence& e)
	{
		ObserveEvidence(f,e);
		const auto unique_vars{ UniqueVars(f)};
		VariableElimination(f, Difference(unique_vars, { v }).values);
		auto m = ComputeJointDistribution(f);
		m.Normalize();
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

	// Set up the adjacency matrix
	//   SetUpAdjacencyMatrix(v,f) creates an undirected graph. If the original graph was a
	//   directed graph, then the resulting graph will be moralized, i.e. it will have edges
	//   between the parents of all children
	//
	//   v is list of all variables in the adjacency matrix
	//   f list of factors containing the variables
	std::vector<std::vector<uint32_t>> SetUpAdjacencyMatrix(const std::vector<uint32_t>& v, const std::vector<Factor>& f) {
		//	cardinality of edges matrix is |v|*|v|
		std::vector<std::vector<uint32_t>> edges(v.size(), std::vector<uint32_t>(v.size(), 0));
      // go through all factors and connect all variables in a factor
		for (size_t i = 0; i < f.size(); ++i) {
			for (size_t j = 0; j < f[i].Var().size(); ++j) {
				for (size_t k = 0; k < f[i].Var().size(); ++k) {
					edges[f[i].Var(j)][f[i].Var(k)] = 1;
				}
			}
		}
		return edges;
	}

	//function F = NormalizeFactorValues( F )
	void NormalizeFactorValues(std::vector<Factor>& f) {
		//  for i=1:length(F)
		std::for_each(f.begin(), f.end(), [](auto& factor) {factor.Normalize(); });
	}

	std::ostream& operator<<(std::ostream& out, const Factor& v) {
		out << "var=" << v.Var() << "card=" << v.Card() << "val=" << v.Val() << std::endl;
		return out;
	}
}
