#include "influence_diagram.h"
#include "utils.h"

namespace Bayes {

	// function EU = SimpleCalcExpectedUtility(I)
	//
	// Inputs: An influence diagram, I (as described in the writeup).
	//         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
	//              the child variable = D.var(1)
	//         I.DecisionFactors = factor for the decision node.
	//         I.UtilityFactors = list of factors representing conditional utilities.
	// Return Value: the expected utility of I
	// Given a fully instantiated influence diagram with a single utility node and decision node,
	// calculate and return the expected utility.  Note - assumes that the decision rule for the 
	// decision node is fully assigned.
	//
	double SimpleCalcExpectedUtility(const InfluenceDiagram id) {
		// In this function, we assume there is only one utility node.
		//	F = [I.RandomFactors I.DecisionFactors];
		auto f{ id.random_factors };
		f.insert(f.end(), id.decision_factors.begin(), id.decision_factors.end());
		//	U = I.UtilityFactors(1);
		auto u = id.utility_factors[0]; // only one utility
		f.push_back(u);

		//	List of all variables
		//	V = unique([F(:).var]);
		const auto v{ UniqueVars(f) };

		// eliminate all variables
		const auto end_factors = VariableElimination(f, v);
		const auto eu = end_factors[0].Val(0);
		return eu;
	}	//end

	// function EUF = CalculateExpectedUtilityFactor(I)
	//
	// Inputs: An influence diagram I with a single decision node and a single utility node.
	//         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
	//              the child variable = D.var(1)
	//         I.DecisionFactors = factor for the decision node.
	//         I.UtilityFactors = list of factors representing conditional utilities.
	// Return value: A factor over the scope of the decision rule D from I that
	// gives the conditional utility given each assignment for D.var
	//
	// Note - We assume I has a single decision node and utility node.
	Factor CalculateExpectedUtilityFactor(const InfluenceDiagram id) {
		//	F = I.RandomFactors;
		auto f{ id.random_factors };
		// In this function, we assume there is only one utility node.
		assert(("only one utility factor supported", id.utility_factors.size() == 1));
		auto u = id.utility_factors[0];
		f.push_back(u);

		//	List of all random variables = List of all variables - decision variable (expect only one)
		uint32_t d = id.decision_factors[0].Var(0);
		std::vector<uint32_t> x;
		for (const auto& factor : id.random_factors)
		{
			x.insert(x.end(), factor.Var().begin(), factor.Var().end());
		}
		std::sort(x.begin(), x.end());
		const auto last = std::unique(x.begin(), x.end());
		x.erase(last, x.end());
		const auto diff{ Difference(x, std::vector<uint32_t>{d}) };
		x = diff.values;

		const auto d_var = id.decision_factors[0].Var(); // only one decision
		std::vector<uint32_t> d_var_parents(d_var.begin() + 1, d_var.end()); // parents have index 1..
		
		const auto x_minus_parents_of_d = Difference(x, d_var_parents);
		const auto factors = VariableElimination(f, x_minus_parents_of_d.values);
		const auto joint = ComputeJointDistribution(factors);
		return joint;
	}

	// function [MEU OptimalDecisionRule] = OptimizeMEU( I )
	//
	// Inputs: An influence diagram I with a single decision node and a single utility node.
	//         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
	//              the child variable = D.var(1)
	//         I.DecisionFactors = factor for the decision node.
	//         I.UtilityFactors = list of factors representing conditional utilities.
	// Return value: the maximum expected utility of I and an optimal decision rule 
	// (represented again as a factor) that yields that expected utility.
	// The order of var in the optimal decison rule is the same as in the internally calculated expected utility factor

	OptimizeInfluenceDiagramResult OptimizeMEU(InfluenceDiagram id) 
	{
		// We assume I has a single decision node.
		// You may assume that there is a unique optimal decision.
		// D = I.DecisionFactors(1);
		assert(("only one decision factor supported", id.decision_factors.size() == 1));
		const auto d = id.decision_factors[0];

		// MEU = 0;
		double meu{ 0.0 };
		// OptimalDecisionRule = struct('var', [], 'card', [], 'val', []);
		Factor odr{ d.Var(), d.Card(), std::vector<double>(d.Val().size())}; 

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 1.  It is probably easiest to think of two cases - D has parents and D 
		//     has no parents.
		// 2.  You may find the Matlab/Octave function setdiff useful.
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
		const auto euf = CalculateExpectedUtilityFactor(id);
		if (d.Var().size() == 1) { // decision variable has no parents
			odr = euf;  // copy var and card from euf
			// calc values
			std::vector<double> val(d.Val().size(), 0.0); 
			// all 0 except the entry that corresponds to the max value of the expected utility factor
			auto it = std::max_element(euf.Val().begin(), euf.Val().end());
			meu = *it;
			val[it - euf.Val().begin()] = 1.0; // best decisision
			odr.SetVal(val);
			return { meu, odr };
		}

		// decision variable has parents

		//  DIndexInEUF = find(EUF.var == D);
		const auto& it = std::find(euf.Var().begin(), euf.Var().end(), d.Var(0)); // expect decision variable as first entry in d
		assert(("d must be in EUF", it != euf.Var().end()));
		const auto d_index_in_euf = std::distance(euf.Var().begin(), it);

		//  D.card = EUF.card( dIndexInEUF );
		const auto d_card = euf.Card(d_index_in_euf);

		const auto parents_diff_result = Difference(euf.Var(), { d.Var(0) });
		const auto parents_var{ parents_diff_result.values };
		const auto mapP2Euf{ parents_diff_result.left_indices };

		// Parents is a dummy factor to create parent assignments
		std::vector<uint32_t> parents_card(parents_var.size(), 0);
		for (size_t i = 0; i < parents_var.size(); ++i) {
			parents_card[i] = euf.Card(mapP2Euf[i]);

		}
		std::vector<double> parents_val(std::accumulate(parents_card.begin(), parents_card.end(), 1, std::multiplies<uint32_t>()), 0.0);
		Factor parents{ parents_var, parents_card, parents_val };

		// create mapping between vars in euf and decision rule
		SetOperationResult<uint32_t> mapping = Union(euf.Var(), d.Var());
		const auto sorted_vars = mapping.values;
		const auto mapEuf = mapping.left_indices;
		const auto mapD = mapping.right_indices;

		// For each assignment of parents ... determine best decision
		// for i=1:length(Parents.val)
		for (size_t i = 0; i < parents_val.size(); ++i) {
			const auto a_pa = parents.IndexToAssignment(i);
			std::vector<uint32_t> a_euf(euf.Var().size(), 0);
			for (size_t k = 0; k < parents_var.size(); ++k) {
				a_euf[mapP2Euf[k]] = a_pa[k];
			}
			// test each decision
			double max_value{ 0.0 };
			size_t idx_euf_max_value{ 0 };
			// for j=1:d.card
			for (uint32_t j = 0; j < d_card; ++j) {
				a_euf[d_index_in_euf] = j;
				const auto idx = euf.AssigmentToIndex(a_euf);
				const auto value = euf.Val(idx);
				if (value > max_value) {
					max_value = value;
					idx_euf_max_value = idx;
				}
			} // end
			// set corresponding entry in decison rule
			const auto a_euf_max = euf.IndexToAssignment(idx_euf_max_value);
			std::vector<uint32_t> a_d(d.Var().size(), 0U);
			for (size_t j = 0; j < d.Var().size(); ++j) {
				// a_euf -> a_d
				a_d[mapD[mapEuf[j]]] = a_euf_max[j];
			}
			const auto idx_odr = odr.AssigmentToIndex(a_d);
			odr.SetVal(idx_odr, 1);
		}
		// eliminate all variables
		const auto factor_product = FactorProduct(odr, euf);
		const auto end_factor = VariableElimination(std::vector<Factor>{ factor_product }, euf.Var());
		meu = end_factor[0].Val(0);

		return {meu, odr};
	}
}
