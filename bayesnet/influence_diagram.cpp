#include "bayesnet/influence_diagram.h"
#include "bayesnet/utils.h"

namespace Bayes {

	// eu = SimpleCalcExpectedUtility(id)
	//
	// Takes as input an influence diagram id and returns the expected utility
	// 
	// Calculates the expexted utility given a fully instantiated influence diagram 
	// with a single utility node and decision node.
	//
	// Note: assumes that the decision rule for the decision node is fully assigned
	//
	double SimpleCalcExpectedUtility(const InfluenceDiagram id) 
	{
		
		// create a list of all factors
		auto f{ id.random_factors };
		f.insert(f.end(), id.decision_factors.begin(), id.decision_factors.end());
		// In this function, we assume there is only one utility node.
		auto u = id.utility_factors[0];
		f.push_back(u);

		//	list of all variables
		const auto v{ UniqueVars(f) };

		// eliminate all variables
		VariableElimination(f, v);

		// get expected utility
		const auto eu = f[0].Val(0);
		return eu;
	}

	// euf = CalculateExpectedUtilityFactor(id)
	//
	// Takes an influence diagram and returns a factor over the scope of the 
	// decision rule d that gives the conditional utility given each assignment for d.var
	//
	// Note: Assume id has a single decision node and utility node.
	//       The decision variable must be the first entry in d.var
	//
	Factor CalculateExpectedUtilityFactor(const InfluenceDiagram id) 
	{
		// In this function, we assume there is only one utility node.
		assert(("only one utility factor supported", id.utility_factors.size() == 1));

		// create list of all factors except decision factor
		auto f{ id.random_factors };
		auto u = id.utility_factors[0];
		f.push_back(u);

		// get decision variable
		const auto d = id.decision_factors[0].Var(0); // only one decision supported

		// x are all variables except decision d
		std::vector<uint32_t> x{ UniqueVars(id.random_factors) };
		const auto diff{ Difference(x, std::vector<uint32_t>{d}) };
		x = diff.values;

		// get parents of decision variable
		const auto d_var = id.decision_factors[0].Var(); 
		std::vector<uint32_t> d_var_parents(d_var.begin() + 1, d_var.end()); // parents have index 1..
		
		// eliminate all random variables except the parents of d
		const auto x_minus_parents_of_d = Difference(x, d_var_parents);
		VariableElimination(f, x_minus_parents_of_d.values);
		// the expected utility factor is the joint distribution of the remaining factors
		const auto euf = ComputeJointDistribution(f);
		return euf;
	}

	// (meu, odr) = OptimizeMEU( id )
	//
	// Takes as input an influence diagram id and returns the maximum expected utility and
	// an optimal decision rule odr (represented as a factor) that yields that expected utility.
	// The order of var in the optimal decison rule is the same as in the internally calculated 
	// expected utility factor euf
	//
	OptimizeInfluenceDiagramResult OptimizeMEU(InfluenceDiagram id) 
	{
		// assume id has a single decision node
		// assume that there is a unique optimal decision
		assert(("only one decision factor supported", id.decision_factors.size() == 1));
		const auto d = id.decision_factors[0];

		double meu{ 0.0 };
		Factor odr{ d.Var(), d.Card(), std::vector<double>(d.Val().size())}; 

		const auto euf = CalculateExpectedUtilityFactor(id);

		if (d.Var().size() == 1) 
		{ 
			// decision variable has no parents
			odr = euf;  // copy var and card from euf
			// calc values
			std::vector<double> val(d.Val().size(), 0.0); 
			// all 0 except the entry that corresponds to the max value of the expected utility factor
			auto it = std::max_element(euf.Val().begin(), euf.Val().end());
			meu = *it;
			val[it - euf.Val().begin()] = 1.0; // mark best decisision
			odr.SetVal(val);
			return { meu, odr };
		}

		// decision variable has parents

		// find index of decision variable in expected utility factor
		const auto& it = std::find(euf.Var().begin(), euf.Var().end(), d.Var(0)); // expect decision variable as first entry in d
		assert(("d must be in EUF", it != euf.Var().end()));
		const auto d_index_in_euf = std::distance(euf.Var().begin(), it);

		// get cardinality of decision variable
		const auto d_card = euf.Card(d_index_in_euf);

		// get parents of decision variable
		const auto parents_diff_result = Difference(euf.Var(), { d.Var(0) });
		const auto parents_var{ parents_diff_result.values };
		const auto mapP2Euf{ parents_diff_result.left_indices };

		// create a dummy factor for parents to create parent assignments
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
		for (size_t i = 0; i < parents_val.size(); ++i) {
			const auto a_pa = parents.IndexToAssignment(i);
			std::vector<uint32_t> a_euf(euf.Var().size(), 0);
			for (size_t k = 0; k < parents_var.size(); ++k) {
				a_euf[mapP2Euf[k]] = a_pa[k];
			}
			// test each decision d
			double max_value{ 0.0 };
			size_t idx_euf_max_value{ 0 };
			for (uint32_t d = 0; d < d_card; ++d) {
				a_euf[d_index_in_euf] = d;
				const auto idx = euf.AssigmentToIndex(a_euf);
				const auto value = euf.Val(idx);
				if (value > max_value) {
					max_value = value;
					idx_euf_max_value = idx;
				}
			}
			// set corresponding entry in decison rule
			const auto a_euf_max = euf.IndexToAssignment(idx_euf_max_value);
			std::vector<uint32_t> a_d(d.Var().size(), 0U);
			for (size_t v = 0; v < d.Var().size(); ++v) {
				// a_euf -> a_d
				a_d[mapD[mapEuf[v]]] = a_euf_max[v];
			}
			const auto idx_odr = odr.AssigmentToIndex(a_d);
			odr.SetVal(idx_odr, 1);
		}
		const auto factor_product = FactorProduct(odr, euf);
		// eliminate all variables
		std::vector<Factor> factors{ factor_product };
		VariableElimination(factors, euf.Var());
		meu = factors[0].Val(0);

		return {meu, odr};
	}
}
