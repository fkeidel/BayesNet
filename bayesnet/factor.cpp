// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#include "bayesnet/factor.h"
#include <numeric>
#include <iostream>
#include <cassert>

namespace Bayes 
{

	Factor::Factor(const std::vector<uint32_t>& var, const std::vector<uint32_t>& card, const std::vector<double>& val) :
		var_(var), card_(card), val_(val)
	{
		assert(("card size must be qual to z size", card.size() == var.size()));
	}

	// index = AssignmentToIndex(assignment) 
	// 
	// Convert assignment to index
	//
	// Converts an assignment over variables to an index into the .val vector
	//   
	std::size_t Factor::AssigmentToIndex(const std::vector<uint32_t>& assignment) const
	{
		std::vector<uint32_t> card{ 1 };
		std::copy(card_.begin(), card_.end() - 1, std::back_inserter(card));
		std::vector<uint32_t> intervals(card_.size(), 0);
		std::partial_sum(card.begin(), card.end(), intervals.begin(), std::multiplies<uint32_t>());
		return std::inner_product(intervals.begin(), intervals.end(), assignment.begin(), 0);
	}

	// assignment = IndexToAssignment(index) 
	// 
	// Convert index to variable assignment
	//
	// Converts an index into the .val vector into an assignment over variables 
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

	double Factor::GetValueOfAssignment(const std::vector<uint32_t>& assignment, const std::vector<uint32_t>& order) const
	{
		const auto intersection_result = Intersection(var_, order);
		assert(("order must contain var", intersection_result.values.size() == var_.size()));

		const auto map_var = intersection_result.left_indices;
		const auto map_order = intersection_result.right_indices;
		std::vector<uint32_t> new_a(var_.size(), 0U);
		for (uint32_t i = 0; i < var_.size(); ++i) {
			new_a[map_var[i]] = assignment[map_order[i]];
		}
		return val_[AssigmentToIndex(new_a)];
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

	// CPD(y)  
	// 
	// Create CPD from factor
	// 
	// Reorder the var, card and val fields of the factor so that the last 
	// variable is the child variable.
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

	// Normalize()
	// 
	// Normalizes the values to sum to 1
	//
	void Factor::Normalize() 
	{
		const auto sum = std::accumulate(val_.begin(), val_.end(), 0.0);
		std::for_each(val_.begin(), val_.end(), [sum](auto& val) { val /= sum; });
	}

	// Marginalize(z)
	// 
	// Sums given variables z out of a factor.
	// 
	// Computes the factor with the variables in z summed out. 
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

	// f = MaxMarginalize(z) 
	// 
	// Takes the max of given variables when marginalizing out of a factor.
	// 
	// Takes in a factor and a set of variables to marginalize out. 
	// For each assignment to the remaining variables, it finds the maximum
	// factor value over all possible assignments to the marginalized variables.
	// The resultant factor should have at least one variable remaining.
	//
	Factor Factor::MaxMarginalize(const std::vector<uint32_t>& z) const
	{
		// check for empty factor or variable list
		if (var_.empty() || z.empty()) return *this;

		// construct the output factor over A.var \ v (the variables in A.var that are not in v)
		SetOperationResult<uint32_t> diff = Difference(var_, z);
		const auto& new_var = diff.values;
		const auto& map_new_var = diff.left_indices;

		// check for empty resultant factor
		assert(("resultant f is empty", !new_var.empty()));

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

	// ObserveEvidence(e)
	// 
	// Modify the factor given some evidence.
	// 
	// Sets all entries in the factor, that are not consistent with 
	// the evidence e, to zero. 
	// e is a vector of variable/value pairs. 
	//
	// Note: does not normalize the factor
	//
	void Factor::ObserveEvidence(const Evidence& e, bool marginalize)
	{
		//  Iterate through all evidence variable/value pairs
		for (const auto& evidence : e) {
			uint32_t e_var = evidence.first;
			uint32_t e_val = evidence.second;

			// does this factor contain the evidence variable?
			const auto& it = std::find(var_.begin(), var_.end(), e_var);
			if (it != var_.end()) 
			{
				//	indx = index of evidence variable in this factor
				const auto indx = std::distance(var_.begin(), it);

				//	check validity of evidence
				assert(("Invalid evidence", e_val < card_[indx]));

				//	adjust the factor to account for observed evidence
				//	for each value
				for (size_t i = 0; i < val_.size(); ++i) {
					// get assignment for this index
					const auto a = IndexToAssignment(i);
					// if assignment is not consistet with the evidence, set value to 0
					if (a[indx] != e_val) 
					{
						SetVal(i, 0.0);
					}
				} // end for k = 1:length(F(j).val)

				if (std::all_of(val_.begin(), val_.end(), [](double val) {return val == 0.0; }))
					std::cout << "Warning: all values in the f are 0" << std::endl;

				if (marginalize)
				{
					*this = Marginalize({ e_var });
				}
			} 
		}
	}

	// c = FactorArithmetic(a,b,op)
	// 
	// Base method to do factor calculation
	// 
	// Applies element-wise operator op to corrsponding values in factors a and b 
	//
	Factor FactorArithmetic(const Factor& a, const Factor& b, const FactorValueOp& op)
	{
		// check for empty factors
		if (a.IsEmpty()) return b;
		if (b.IsEmpty()) return a;

		// check that variables in both A and B have the same cardinality
		SetOperationResult<uint32_t> intersection = Intersection(a.Var(), b.Var());
		if (!intersection.values.empty()) 
		{
			// assert that a and b have at least 1 variable in common
			const auto& iA = intersection.left_indices;
			const auto& iB = intersection.right_indices;
			for (size_t i = 0; i < intersection.values.size(); ++i)
			{
				assert(("Dimensionality mismatch in factors", a.Card(iA[i]) == b.Card(iB[i])));
			}
		}

		// Set the variables of c and construct the mapping between variables in a and b and variables in c.
		// In the code below, we have that
		//	
		//	map_a_to_c(i) = j, if and only if, a.var(i) == c.var(j)
		// and similarly
		// map_b_to_c(i) = j, if and only if, b.var(i) == c.var(j)
		//	
		// For example, if a.var = [2 0 3], b.var = [3 4], and c.var = [0 2 3 4],
		//	then, map_a_to_c = [1 0 2] and map_b_to_c = [3 4]; 
		// map_a_to_c(0) == 1 because a.var(0) == 2 and c.var(1) = 2, so a.var(0) == c.var(1).

		SetOperationResult<uint32_t> union_result = Union(a.Var(), b.Var());
		const auto& c_var = union_result.values;
		const auto& map_a_to_c = union_result.left_indices;
		const auto& map_b_to_c = union_result.right_indices;

		// Set the cardinality of variables in c
		std::vector<uint32_t> c_card(c_var.size(), 0);
		for (size_t i = 0; i < map_a_to_c.size(); ++i) 
		{
			c_card[map_a_to_c[i]] = a.Card(i);
		}
		for (size_t i = 0; i < map_b_to_c.size(); ++i) 
		{
			c_card[map_b_to_c[i]] = b.Card(i);
		}

		Factor c{ union_result.values,c_card, {} };

		// create assignments of c
		const auto assignments_size = std::accumulate(c_card.begin(), c_card.end(), 1, std::multiplies<uint32_t>());
		std::vector<std::vector<uint32_t>> assignments(assignments_size);
		for (size_t i = 0; i < assignments_size; ++i)
		{
			assignments[i] = c.IndexToAssignment(i);
		}

		std::vector<size_t> index_a(assignments_size);
		for (size_t i = 0; i < assignments_size; ++i) 
		{
			std::vector<uint32_t> assignment_a(a.Var().size());
			for (size_t j = 0; j < a.Var().size(); ++j)
			{
				assignment_a[j] = assignments[i][map_a_to_c[j]];
			}
			index_a[i] = a.AssigmentToIndex(assignment_a);
		}

		std::vector<size_t> index_b(assignments_size);
		for (size_t i = 0; i < assignments_size; ++i) 
		{
			std::vector<uint32_t> assignment_b(b.Var().size());
			for (size_t j = 0; j < b.Var().size(); ++j)
			{
				assignment_b[j] = assignments[i][map_b_to_c[j]];
			}
			index_b[i] = b.AssigmentToIndex(assignment_b);
		}

		// apply the element-wise operator op to corrsponding entries in a and b
		std::vector<double> c_values;
		for (size_t i = 0; i < assignments_size; ++i) 
		{
			c_values.push_back(op(a.Val(index_a[i]), b.Val(index_b[i])));
		}

		c.SetVal(c_values);

		return c;
	}

	// c = FactorProduct(a,b) 
	// 
	// Computes the product of two factors a and b
	//
	Factor FactorProduct(const Factor& a, const Factor& b) 
	{
		return FactorArithmetic(a, b, FactorValueMultiply{});
	}

	//	FactorSum(a,b)
	// 
	// Computes the sum of two factors  a and b
	//
	Factor FactorSum(const Factor& a, const Factor& b) 
	{
		return FactorArithmetic(a, b, FactorValueAdd{});
	}


	// ObserveEvidence(f, e) 
	// 
	// Modify a vector of factors given some evidence.
	// 
	// Sets all entries in the vector of factors f, that are not consistent with 
	// the evidence e, to zero. 
	// e is a vector of variable/value pairs. 
	//
	// Note: does not normalize the factor
	//
	void ObserveEvidence(std::vector<Factor>& f, const Evidence& e)
	{
		//    Iterate through the factors
		for (auto& factor : f)
		{
			factor.ObserveEvidence(e);
		}
	}

	// EliminateVar(f,e,z)
	// 
	// Elimiates a variable z from a list of factors f, given the adjacency matrix for the variables
	// 
	//	f = list of factors
	//	e = adjacency matrix for variables
	//	z = variable to eliminate
	//
	void EliminateVar(std::vector<Factor>& f, std::vector<std::vector<uint32_t>>& e, uint32_t z)
	{
		//	Index of factors to multiply (they contain z)
		std::vector<size_t> use_factors;

		// Union of scopes of factors to multiply
		std::vector<uint32_t> scope;
		// go through all factors
		for (size_t i = 0; i < f.size(); ++i) {
			// if the variables in the factor contains z
			if (std::any_of(f[i].Var().begin(), f[i].Var().end(), [z](uint32_t var) {return var == z; })) 
			{
				// add the factor to the list of factors to multiply
				use_factors.push_back(i);
				// add scope of the factor to the scope of the list of factors to multiply
				scope = Union(scope, f[i].Var()).values;
			}
		}

		//	update edge map
		//	new edges represent the induced edges for the variable elimination graph.
		for (size_t i = 0; i < scope.size(); ++i) {
			for (size_t j = 0; j < scope.size(); ++j) {
				if (i != j) {
					e[scope[i]][scope[j]] = 1;
					e[scope[j]][scope[i]] = 1;
				}
			}
		}

		//	Remove all adjacencies for the variable to be eliminated
		for (size_t j = 0; j < e[z].size(); ++j) {
			e[z][j] = 0;
		}
		for (size_t i = 0; i < e.size(); ++i) {
			e[i][z] = 0;
		}

		// non_use_factors = list of factors that don't contain z
		std::vector<size_t> range_f(f.size(), 0);
		std::iota(range_f.begin(), range_f.end(), 0);
		std::vector<size_t> non_use_factors = Difference(range_f, use_factors).values;

		//	new_f = list of factors we will return
		std::vector<Factor> new_f(non_use_factors.size());

		// copy the non_use_factors to the reuslt
		for (size_t i = 0; i < non_use_factors.size(); ++i) 
		{
			new_f[i] = f[non_use_factors[i]];
		}

		//	Multiply factors which involve z to get a new factor
		Factor new_factor{};
		for (size_t i = 0; i < use_factors.size(); ++i) 
		{
			new_factor = FactorProduct(new_factor, f[use_factors[i]]);
		}

		// eliminate z
		new_factor = new_factor.Marginalize({ z });
		new_f.push_back(new_factor);
		f = new_f;
	}

	//	VariableElimination(f, z)
	// 
	// Runs the Variable Elimination algorithm
	// 
	// VariableElimination takes in a list of factors f and a list z of variables to eliminate
	//	and returns the resulting factor after running sum-product to eliminate
	//	the given variables.
	//	
	//	
	//	f = list of factors
	//	z = list of variables to eliminate
	//
	void VariableElimination(std::vector<Factor>& f, const std::vector<uint32_t>& z)
	{
		//	sorted list of all variables
		const auto v{ UniqueVars(f) };

		// set up the adjacency matrix.
		auto edges{ SetUpAdjacencyMatrix(v,f) };

		for (size_t i = 0; i < z.size(); ++i)
		{
			// use the Min-Neighbors heuristic to eliminate the variable that has
			// the smallest number of edges connected to it in each cycle
			const auto best_variable = MinNeighbor(edges, z);
			EliminateVar(f, edges, best_variable);
		} //end
	}

	// best_variable = MinNeighbour(edges, z)
	//
	// Min-Neighbours heuristic
	// 
	// Returns the variable in z that has the smallest number of edges connected to it
	//
	uint32_t MinNeighbor(std::vector<std::vector<uint32_t>>& edges, const std::vector<uint32_t >& z)
	{
		uint32_t best_variable{ 0 };
		uint32_t best_score{ std::numeric_limits<uint32_t>::max() };
		for (size_t i = 0; i < z.size(); ++i) {
			const auto idx = z[i];
			//	the score is the sum of '1's for a variable in the corresponding line in the edges matrix
			const auto score = std::accumulate(edges[idx].begin(), edges[idx].end(), 0U);
			//	selects the variable with the smallest score
			if ((score > 0) && (score < best_score)) {
				best_score = score;
				best_variable = idx;
			}
		}
		return best_variable;
	}

	// joint = ComputeJointDistribution(f) 
	// 
	// Computes the joint distribution defined by a set of factors f by multiplying 
	// all factors
	//
	Factor ComputeJointDistribution(const std::vector<Factor>& f) {
		Factor joint{};
		for (size_t i = 0; i < f.size(); ++i) {
			joint = FactorProduct(joint, f[i]);
		}
		return joint;
	}

	// m = SimpleComputeMarginal(v,f,e) 
	// 
	// Computes the marginal over variables v in the distribution induced by the set of factors f, 
	// given evidence e. The function computes the joint distribution of the factors
	// and then marginalzes out all variables except the variables contained in v
	// 
	// Note: This method is inefficient, if you have a many factors, or if you want to 
	//       calculate more than one marginal from a set of factors. In these cases
	//       use more efficient algorithms like VariableElimination or CliqueTree
	//
	// m is a factor containing the marginal over variables v
	// v is a vector containing the variables in the marginal e.g. [1 2 3] for
	//   the variables 1,2,3
	// f is a vector of factors defining the distribution
	// e is vector of variable/value pairs
	//
	Factor SimpleComputeMarginal(const std::vector<uint32_t>& v, std::vector<Factor>& f, const Evidence& e) {
		// Check for empty factor list
		if (f.empty()) return {};
		ObserveEvidence(f, e);
		Factor joint = ComputeJointDistribution(f);
		Factor m = joint.Marginalize(Difference(joint.Var(), v).values);
		m.Normalize();
		return m;
	}

	// m = VariableEliminationComputeExactMarginalBP(v,f,e)
	// 
	// Computes the marginal over variables v in the distribution induced by the set of factors f, 
	// given evidence e. The function eliminates one variable after the other until only the
	// variables in v remain and then normalizes the result
	// 
	// m is a factor containing the marginal over variables v
	// v is a vector containing the variables in the marginal e.g. [1 2 3] for
	//   the variables 1,2,3
	// f is a vector of factors defining the distribution
	// e is vector of variable/value pairs
	//
	Factor VariableEliminationComputeExactMarginalBP(const uint32_t v, std::vector<Factor>& f, const Evidence& e)
	{
		ObserveEvidence(f,e);
		const auto unique_vars{ UniqueVars(f)};
		VariableElimination(f, Difference(unique_vars, { v }).values);
		auto m = ComputeJointDistribution(f);
		m.Normalize();
		return m;
	}

	// v = UniqueVars(f) 
	// 
	//	Get a vector v of unique variables from a vector f of factors
	//	The returned variables will be ordered
	//
	// f vector of factors containing the variables
	//
	std::vector<uint32_t> UniqueVars(std::vector<Factor> f) {
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

	// edges = SetUpAdjacencyMatrix(v,f) 
	// 
	// Set up the adjacency matrix
	// 
	//	The function creates an undirected graph represented as adjacency matrix. 
	//	If the original graph was a directed graph, then the resulting graph will be moralized, 
	// i.e. it will have edges between the parents of all children
	//
	// v vector of all variables in the adjacency matrix
	// f vector of factors containing the variables
	//	edges matrix of edges between variables, where an edge between variables i and j 
	//       is represented by edges[i][j] == 1
	//
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

	// NormalizeFactorValues(f)
	// 
	// Normalizes all factors in f
	//
	void NormalizeFactorValues(std::vector<Factor>& f) {
		std::for_each(f.begin(), f.end(), [](auto& factor) {factor.Normalize(); });
	}

	std::ostream& operator<<(std::ostream& out, const Factor& f) {
		out << "var=" << f.Var() << "card=" << f.Card() << "val=" << f.Val() << std::endl;
		return out;
	}
}
