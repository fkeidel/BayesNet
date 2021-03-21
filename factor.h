#ifndef FACTOR_H
#define FACTOR_H

#include<vector>
#include<cstdint>
#include<iostream>
#include "utils.h"

namespace Bayes {

	class Factor {
	public:
		Factor() = default;
		Factor(const std::vector<uint32_t>& var, const std::vector<uint32_t>& card, const std::vector<double>& val);

		std::size_t AssigmentToIndex(const std::vector<uint32_t>& assignment)  const;
		std::vector<uint32_t> IndexToAssignment(size_t index) const;

		double GetValueOfAssignment(const std::vector<uint32_t>& assignment);
		void SetValueOfAssignment(const std::vector<uint32_t>& assignment, double value);

		bool operator==(const Factor& rhs) const
		{
			return (var_ == rhs.var_)
				&& (card_ == rhs.card_)
				&& (val_ == rhs.val_);
		}

		bool operator!=(const Factor& rhs) const
		{
			return !operator==(rhs);
		}

		bool IsEmpty() const {
			return var_.empty();
		}

		const std::vector<uint32_t>& Var() const { return var_; }
		const std::vector<uint32_t>& Card() const { return card_; }
		const std::vector<double>& Val() const { return val_; }

		void SetVal(const std::vector<double>& val);
		void SetVal(size_t index, double val);


	private:
		std::vector<uint32_t> var_;		// list of variable ids
		std::vector<uint32_t> card_;	// list of cardinalities of variables
		std::vector<double> val_;		// list of values
	};

	Factor FactorProduct(const Factor& a, const Factor& b);
	Factor FactorMarginalization(const Factor& a, const std::vector<uint32_t>& var);
	void ObserveEvidence(std::vector<Factor>& f, const std::vector<std::pair<uint32_t, uint32_t>>& e);
	void EliminateVar(std::vector<Factor>& f, std::vector<std::vector<uint32_t>>& e, uint32_t z);
	std::vector<Factor> VariableElimination(std::vector<Factor>& f, std::vector<uint32_t>& z);

	std::ostream& operator<<(std::ostream& out, const Factor& v);
}

#endif // FACTOR_H