#include "factor.h"
#include "factor.h"
#include "factor.h"
#include "factor.h"
#include "utils.h"
#include <numeric>
#include <iterator>
namespace Bayes {
	Factor::Factor(const std::vector<uint32_t>& var, const std::vector<uint32_t>& card, const std::vector<double>& val) :
		var_(var), card_(card), val_(val)
	{
	}

	std::size_t Factor::AssigmentToIndex(const std::vector<uint32_t>& assignment) const
	{
		std::vector<uint32_t> intervals(card_.size(), 0);
		std::vector<uint32_t> card{ 1 };
		std::copy(card_.begin(), card_.end() - 1, std::back_inserter(card));
		std::partial_sum(card.begin(), card.end(), intervals.begin(), std::multiplies<uint32_t>());
		return std::inner_product(intervals.begin(), intervals.end(), assignment.begin(), 0);
	}

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

	double Factor::GetValueOfAssignment(const std::vector<uint32_t>& assignment)
	{
		return val_[AssigmentToIndex(assignment)];
	}

	void Factor::SetValueOfAssignment(const std::vector<uint32_t>& assignment, double value)
	{
		val_[AssigmentToIndex(assignment)] = value;
	}

	void Factor::SetVal(const std::vector<double>& val)
	{
		val_ = val;
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
		std::vector<uint32_t> c_card(c_var.size(),0);  // C.card = zeros(1, length(C.var));
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

	Factor FactorMarginalization(const Factor& a, std::vector<uint32_t>& v)
	{
		// Check for empty factor or variable list
		if (a.Var().empty() || v.empty()) return a;

		// Construct the output factor over A.var \ v (the variables in A.var that are not in v)
		SetOperationResult<uint32_t> diff = Difference(a.Var(), v);
		const auto& b_var = diff.values;
		const auto& mapB = diff.left_indices;

		// Check for empty resultant factor
		if (b_var.empty()) return {};

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


}