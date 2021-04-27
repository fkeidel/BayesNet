// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#ifndef FACTOR_H
#define FACTOR_H

#include<vector>
#include<cstdint>
#include<iostream>
#include "utils.h"

namespace Bayes {

	using Evidence = std::vector<std::pair<uint32_t, uint32_t>>;


	struct FactorValueOp {
		virtual double operator()(const double lhs, const double rhs) const = 0;
	};

	struct FactorValueMultiply : FactorValueOp {
		double operator()(const double lhs, const double rhs) const override {
			return lhs * rhs;
		}
	};

	struct FactorValueAdd : FactorValueOp {
		double operator()(const double lhs, const double rhs) const override {
			return lhs + rhs;
		}
	};

	class Factor {
	public:

		Factor() = default;
		Factor(const std::vector<uint32_t>& var, const std::vector<uint32_t>& card, const std::vector<double>& val);

		std::size_t AssigmentToIndex(const std::vector<uint32_t>& assignment)  const;
		std::vector<uint32_t> IndexToAssignment(size_t index) const;

		double GetValueOfAssignment(const std::vector<uint32_t>& assignment) const;
		// alias
		double Val(const std::vector<uint32_t>& assignment) const { return GetValueOfAssignment(assignment); }
		double operator()(const std::vector<uint32_t>& assignment) const {
			return GetValueOfAssignment(assignment);
		}

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
		uint32_t Var(size_t index) const { return var_[index]; }

		const std::vector<uint32_t>& Card() const { return card_; }
		uint32_t Card(size_t index) const { return card_[index]; }

		const std::vector<double>& Val() const { return val_; }
		double Val(size_t index) const { return val_[index]; }

		void SetVar(const std::vector<uint32_t>& val);

		void SetCard(const std::vector<uint32_t>& card);
		void SetCard(size_t index, uint32_t card);

		void SetVal(const std::vector<double>& val);
		void SetVal(size_t index, double val);

		Factor CPD(uint32_t y);
		void Normalize();

		Factor Marginalize(const std::vector<uint32_t>& var) const;
		Factor MaxMarginalize(const std::vector<uint32_t>& var) const;

		void ObserveEvidence(const Evidence& e);

	private:
		std::vector<uint32_t> var_;		// list of variable ids
		std::vector<uint32_t> card_;	// list of cardinalities of variables
		std::vector<double> val_;		// list of values
	};

	Factor FactorProduct(const Factor& a, const Factor& b);
	Factor FactorSum(const Factor& a, const Factor& b);
	Factor FactorArithmetic(const Factor& a, const Factor& b, const FactorValueOp& op);
	void ObserveEvidence(std::vector<Factor>& f, const Evidence& e);
	void EliminateVar(std::vector<Factor>& f, std::vector<std::vector<uint32_t>>& e, uint32_t z);
	void VariableElimination(std::vector<Factor>& f, const std::vector<uint32_t>& z);
	Factor ComputeJointDistribution(const std::vector<Factor>& f);
	Factor SimpleComputeMarginal(const std::vector<uint32_t>& v, std::vector<Factor>& f, const Evidence& e);
	Factor VariableEliminationComputeExactMarginalBP(const uint32_t v, std::vector<Factor>& f, const Evidence& e);
	std::vector<uint32_t> UniqueVars(std::vector<Factor> f);
	std::vector<std::vector<uint32_t>> SetUpAdjacencyMatrix(const std::vector<uint32_t>& v, const std::vector<Factor>& f);
	void NormalizeFactorValues(std::vector<Factor>& f);
	
	std::ostream& operator<<(std::ostream& out, const Factor& v);
}

#endif // FACTOR_H