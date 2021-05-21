// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#include "bayesnet/sampling.h"
#include "bayesnet/utils.h"
#include <numeric>
#include <set>

namespace Bayes {

	// log_prob = LogProbOfJointAssignment(f, assignment)
	//
	//  Returns the log probability of an assignment in a distribution defined by factors f
	double LogProbOfJointAssignment(const std::vector<Factor>& factors, const std::vector<uint32_t>& assignment) {
		// work in log-space to prevent underflow
		//logp = 0.0;
		double log_prob = 0.0;
		std::vector<uint32_t> order(assignment.size(), 0U);
		std::iota(order.begin(), order.end(), 0U);
		//for i = 1:length(F)
		for (uint32_t i = 0; i < factors.size(); ++i) {
			//    logp = logp + log(GetValueOfAssignment(F(i), A, 1:length(A)));
			log_prob += std::log(factors[i].GetValueOfAssignment(assignment, order));
		}//end
		return log_prob;
	}

	//BLOCKLOGDISTRIBUTION
	//
	//   LogBS = BlockLogDistribution(V, G, F, A) returns the log of a
	//   block-sampling array (which contains the log-unnormalized-probabilities of
	//   selecting each label for the block), given variables V to block-sample in
	//   network G with factors F and current assignment A.  Note that the variables
	//   in V must all have the same dimensionality.
	//
	//   Input arguments:
	//   V -- an array of variable names
	//   G -- the graph with the following fields:
	//     .names - a cell array where names{i} = name of variable i in the graph 
	//     .card - an array where card(i) is the cardinality of variable i
	//     .edges - a matrix such that edges(i,j) shows if variables i and j 
	//              have an edge between them (1 if so, 0 otherwise)
	//     .var2factors - a cell array where var2factors{i} gives an array where the
	//              entries are the indices of the factors including variable i
	//   F -- a struct array of factors.  A factor has the following fields:
	//       F(i).var - names of the variables in factor i
	//       F(i).card - cardinalities of the variables in factor i
	//       F(i).val - a vectorized version of the CPD for factor i (raw probability)
	//   A -- an array with 1 entry for each variable in G s.t. A(i) is the current
	//       assignment to variable i in G.
	//
	//   Each entry in LogBS is the log-probability that that value is selected.
	//   LogBS is the P(V | X_{-v} = A_{-v}, all X_i in V have the same value), where
	//   X_{-v} is the set of variables not in V and A_{-v} is the corresponding
	//   assignment to these variables consistent with A.  In the case that |V| = 1,
	//   this reduces to Gibbs Sampling.  NOTE that exp(LogBS) is not normalized to
	//   sum to one at the end of this function (nor do you need to worry about that
	//   in this function).
	//
	//
	//function LogBS = BlockLogDistribution(V, G, F, A)
	std::vector<double> BlockLogDistribution(std::vector<uint32_t> var, Graph graph, std::vector<Factor> factors, std::vector<uint32_t> assignment) 
	{
		// d is the dimensionality of all the variables we are extracting
		//d = G.card(V(1));
		const auto d = graph.card[var[0]];
		//if length(unique(G.card(V))) ~= 1
		//    disp('WARNING: trying to block sample invalid variable set');
		//    return;
		//end
		assert(("all vars in var must have same cardinality",
			std::all_of(graph.card.cbegin(), graph.card.cend(), [d](uint32_t card) { return card == d; })));

		//LogBS = zeros(1, d);
		std::vector<double> log_bs(d, 0.0);
		//
		// YOUR CODE HERE
		// Compute LogBS by multiplying (adding in log-space) in the correct values from
		// each factor that includes some variable in V.  
		//
		// NOTE: As this is called in the innermost loop of both Gibbs and Metropolis-
		// Hastings, you should make this fast.  You may want to make use of
		// G.var2factors, repmat, unique, and GetValueOfAssignment.
		//
		// Also you should have only ONE for-loop, as for-loops are VERY slow in matlab
		//
		//factors = F(unique([G.var2factors{V}]));
		std::vector<uint32_t> ind_factors_of_v;
		for (uint32_t i = 0; i < var.size(); ++i) {
			ind_factors_of_v.insert(ind_factors_of_v.end(), graph.var2factors[var[i]].begin(), graph.var2factors[var[i]].end());
		}
		Unique(ind_factors_of_v);
		std::vector<Factor> factors_of_v(ind_factors_of_v.size());
		for (uint32_t i = 0; i < factors_of_v.size(); ++i) {
			factors_of_v[i] = factors[ind_factors_of_v[i]];
		}

		//for i = 1:d
		for (uint32_t i = 0; i < d; ++i) {
			//    A(V) = i;
			for (uint32_t v = 0; v < var.size(); ++v) {
				assignment[var[v]] = i;
			}
			//    LogBS(i) = LogProbOfJointAssignment(factors, A);
			log_bs[i] = LogProbOfJointAssignment(factors_of_v, assignment);
		}//end
		//
		// Re-normalize to prevent underflow when you move back to probability space
		//LogBS = LogBS - min(LogBS);
		NormalizeLog(log_bs);
		return log_bs;
	}

	//LogBS = LogBS - min(LogBS);
	void NormalizeLog(std::vector<double>& logs) 
	{
		const auto min_log = *std::min_element(logs.begin(), logs.end());
		std::for_each(logs.begin(), logs.end(), [min_log](auto& val) { val -= min_log; });
	}

	// randsample(V, n, true, distribution) returns a set of n values sampled
	// at random from the integers 1 through V with replacement using distribution
	// 'distribution'
	//
	// replacing true with false causes sampling w/out replacement
	// omitting the distribution causes a default to the uniform distribution
	//
	//function[v] = randsample(vals, numSamp, replace, weightIncrements)
	std::vector<uint32_t> RandSample(std::vector<uint32_t>& vals, uint32_t num_samp, bool replace, std::vector<double> weight_increments, std::minstd_rand&  gen)
	{
		//vals = vals(:);
		uint32_t max_val{ 0U };
		//if(length(vals)==1)
		if (vals.size() == 1)
		{
			//  maxval = vals;
			max_val = vals.front();
			vals.resize(max_val + 1U);
			std::iota(vals.begin(), vals.end(), 0);
			//  vals = 1:maxval;
		}
		else //else
		{
			//  maxval = length(vals);
			max_val = vals.back();
		}//end

		//if(exist('replace','var')~=1)
		//  replace = true;
		//end
		std::vector<double> weights;
		//if(exist('weightIncrements','var')~=1)
		if (weight_increments.empty()) {
			//  weightIncrements = (1/maxval)*ones(maxval,1);
			weight_increments = std::vector<double>(max_val + 1U, 1.0);
			std::for_each(weight_increments.begin(), weight_increments.end(), [max_val](auto& value) { value /= max_val + 1U;});
			// weights = (1 / maxval) : (1 / maxval) : 1;
			weights.resize(max_val + 1U);
		} //else
		else 
		{
			//  weightIncrements = weightIncrements(:)/sum(weightIncrements(:));
			const auto sum = std::accumulate(weight_increments.begin(), weight_increments.end(), 0.0);
			std::for_each(weight_increments.begin(), weight_increments.end(), [sum](auto& val) { val /= sum; });

			//  weights = zeros(size(weightIncrements));
			weights.resize(weight_increments.size());
			//  weights(1) = weightIncrements(1);
			//  for i = 2:length(weightIncrements)
			//    weights(i) = weightIncrements(i)+weights(i-1);
			//  end
		} //end
		std::partial_sum(weight_increments.begin(), weight_increments.end(), weights.begin(), std::plus<double>());
		//weights = [0; weights(:)];
		weights.insert(weights.begin(), 0.0);
		//
		//now do the sampling
		//v = [];
		std::vector<uint32_t> v(num_samp);
		//probs = rand(numSamp,1);
		std::uniform_real_distribution<> uni_real_dist(0.0, 1.0);
		//for i=1:numSamp
		for (uint32_t i = 0; i < num_samp; ++i) {
			double prob = uni_real_dist(gen);
			//  curInd = find((weights(1:end-1)<=probs(i))&(weights(2:end)>=probs(i)));
			const auto it = std::find_if(weights.begin(), weights.end(), [prob](const auto val) { return val > prob;});
			const auto cur_ind = std::distance(weights.begin(), it)-1;
			//  v(end+1)=vals(curInd);
			v[i] = vals[cur_ind];
			//  if(replace~=true)
			if (!replace) {
				//    vals(curInd)=[];
				vals.erase(vals.begin() + cur_ind);
				//    weightIncrements(curInd)=[];
				weight_increments.erase(weight_increments.begin() + cur_ind);
				//    weightIncrements = weightIncrements(:)/sum(weightIncrements(:));
				const auto sum = std::accumulate(weight_increments.begin(), weight_increments.end(), 0.0);
				std::for_each(weight_increments.begin(), weight_increments.end(), [sum](auto& val) { val /= sum; });
				//    weights = zeros(size(weightIncrements));
				weights.resize(weight_increments.size());
				//    for i = 2:length(weightIncrements)
				//      weights(i) = weightIncrements(i)+weights(i-1);
				//    end
				std::partial_sum(weight_increments.begin(), weight_increments.end(), weights.begin(), std::plus<double>());
				weights.insert(weights.begin(), 0.0);
			}//  end
		} //end
		return v;
	}

	// GIBBSTRANS
	//
	//  MCMC transition function that performs Gibbs sampling.
	//  A - The current joint assignment.  This should be
	//      updated to be the next assignment
	//  G - The network
	//  F - List of all factors
	//
	//function A = GibbsTrans(A, G, F)
	std::vector<uint32_t> GibbsTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand& gen)
	{
		std::vector<uint32_t> new_assignment(assignment);
		//for i = 1:length(G.names)
		const auto num_var{ graph.var.size() };
		for (uint32_t i = 0; i < num_var; ++i) {
			//    
			//     YOUR CODE HERE
			//     For each variable in the network sample a new value for it given everything
			//     else consistent with A.  Then update A with this new value for the
			//     variable.  NOTE: Your code should call BlockLogDistribution().
			//     IMPORTANT: you should call the function randsample() exactly once
			//     here, and it should be the only random function you call.
			//    
			//     Also, note that randsample() requires arguments in raw probability space
			//     be sure that the arguments you pass to it meet that criteria
			//    
			//    A(i) = randsample(G.card(i), 1, true, exp(BlockLogDistribution(i, G, F, A)));
			auto dist = BlockLogDistribution({ i }, graph, factors, new_assignment);
			std::for_each(dist.begin(), dist.end(), [](auto& val) { val = std::exp(val);});
			std::vector<uint32_t> vals{ graph.card[i]-1 };
			new_assignment[i] = RandSample(vals, 1, true, dist, gen).front();
			//    
		}//end
		return new_assignment;
	}

	//EXTRACTMARGINALSFROMSAMPLES
	//
	//   ExtractMarginalsFromSamples takes in a probabilistic network G, a list of samples, and a set
	//   of indices into samples that specify which samples to use in the computation of the
	//   marginals.  The marginals are then computed using this subset of samples and returned.
	//
	//   Samples is a matrix where each row is the assignment to all variables in
	//   the network (samples(i,j)=k means in sample i the jth variable takes label k)
	//
	//function M = ExtractMarginalsFromSamples(G, samples, collection_indx)
	std::vector<Factor> ExtractMarginalsFromSamples(
		const Graph& graph, 
		const std::vector<std::vector<uint32_t>>& samples, 
		const std::vector<uint32_t>& collection_indx) 
	{
		//
		//collected_samples = samples(collection_indx, :);
		std::vector<std::vector<uint32_t>> collected_samples(collection_indx.size());
		for (uint32_t i = 0; i < collection_indx.size(); ++i) {
			collected_samples[i] = samples[collection_indx[i]];
		}
		//
		//M = repmat(struct('var', 0, 'card', 0, 'val', []), length(G.names), 1);
		const auto num_var{ graph.var.size() };
		std::vector<Factor> m(num_var);
		//for i = 1:length(G.names)
		for (uint32_t i = 0; i < num_var; ++i) {
			//    M(i).var = i;
			//    M(i).card = G.card(i);
			//    M(i).val = zeros(1, G.card(i));
			m[i].SetVar({ i });
			m[i].SetCard({ graph.card[i] });
			m[i].SetVal(std::vector<double>(graph.card[i]));
		}//end
		//
		//for s=1:size(collected_samples, 1)
		for (uint32_t s = 0; s < collected_samples.size(); ++s) 
		{
			//    sample = collected_samples(s,:);
			const auto& sample{ collected_samples[s] };
			//    for j=1:length(sample)
			for (uint32_t j = 0; j < sample.size(); ++j) 
			{
				// M(j).val(sample(j)) = M(j).val(sample(j)) + 1/size(collected_samples,1);
				double val = m[j].Val(sample[j]) + 1.0 / collected_samples.size();
				m[j].SetVal(sample[j], val);
			}//    end
		}//end
		//
		return m;
	}
	//
	// Returns a matrix that maps edges to a list of factors in which both ends partake
	//
	//function E2F = EdgeToFactorCorrespondence(V, F)
	std::vector<std::vector<std::vector<uint32_t>>> EdgeToFactorCorrespondence(const std::vector<uint32_t>& var, std::vector<Factor> factors)
	{
		//E2F = cell(length(V), length(V));
		std::vector<std::vector<std::vector<uint32_t>>> e2f(var.size(), std::vector<std::vector<uint32_t>>(var.size()));
		//for f = 1:length(F)
		for (uint32_t f = 0; f < factors.size(); ++f) {
			//    for i = 1:length(F(f).var)
			for (uint32_t i = 0; i < factors[f].Var().size(); ++i)
			{
				// for j = i+1:length(F(f).var)
				for (uint32_t j = 0; j < factors[f].Var().size(); ++j)
				{
					// u = F(f).var(i);
					// v = F(f).var(j);
					const auto& u{ factors[f].Var(i) };
					const auto& v{ factors[f].Var(j) };
					// E2F{u,v} = union(E2F{u,v}, f);
					// E2F{v,u} = union(E2F{v,u}, f);
					e2f[u][v].push_back(f);
					e2f[v][u].push_back(f);
				}// end
			}// end
		}//end
		return e2f;
	}

	QList CreateQList(Graph graph, std::vector<Factor> factors, SWVariant variant ) 
	{
		//   Swendsen-Wang computation of Q matrix (contains q_{i,j}'s)
		QList q_list;
		//  E2F = EdgeToFactorCorrespondence(G.names, F);
		const auto e2f = EdgeToFactorCorrespondence(graph.var, factors);
		//  [u, v] = find(G.edges);
		//  q_list = [];   each row in q_list is of the form [node_i, node_j, q_ij]
		//  for i = 1:size([u, v], 1)
		//       For every non-directed edge (don't want to double count)
		//      if u(i) > v(i)
		for (uint32_t i = 0; i < graph.edges.size(); ++i)
		{
			for (uint32_t j = 0; j < graph.edges.size(); ++j)
			{
				if ((i > j) && (graph.edges[i][j] == 1)) {
					// edge_factor = F(E2F{u(i), v(i)}(1));
					const auto& edge_factor = factors[e2f[i][j][0]];
					assert(("cardinalities of variables in edge factor must be equal", edge_factor.Card(0) == edge_factor.Card(1)));
					// q_ij = 0.0;
					double q_ij{ 0.0 };
					// if strcmp(TransName, 'MHSwendsenWang1')
					if (variant == UNIFORM)
					{
						// Specify the q_{i,j}'s for Swendsen-Wang for variant 1
						q_ij = 0.5;
					}
					// elseif strcmp(TransName, 'MHSwendsenWang2')
					else if (variant == BLOCK_SAMPLING)
					{
						// Specify the q_{i,j}'s for Swendsen-Wang for variant 2
						// q_ij = sum(Fij(u, u)) / sum(Fij(u, v))
						//         u               u,v
						double sum_u_fij{ 0.0 };
						double sum_u_v_fij{ 0.0 };
						for (uint32_t u = 0; u < edge_factor.Card(0); ++u) {
							for (uint32_t v = 0; v < edge_factor.Card(1); ++v) {
								sum_u_v_fij += edge_factor({ u,v });
								if (u == v) {
									sum_u_fij += edge_factor({ u,u });
								}
							}
						}
						q_ij = sum_u_fij / sum_u_v_fij;
					}// else
					else {
						// disp('WARNING: unrecognized Swendsen-Wang name');
						std::cout << "WARNING: unrecognized Swendsen-Wang name" << std::endl;
					}// end
					// assert(q_ij >= 0.0 && q_ij <= 1.0);
					assert(("q_ij must be in range [0,1]", q_ij >= 0.0 && q_ij <= 1.0));
					// q_list = [q_list; u(i), v(i), q_ij];
					q_list.push_back({ i,j,q_ij });
				} // end
			}
		} //  end
		//  G.q_list = q_list;
		return q_list;
	}

	//	 MCMCINFERENCE conducts Markov Chain Monte Carlo Inference.
	//  M = MCMCInference(G, F, E, ...) performs inference given a Markov Net or Bayes Net, G, a list
	//  of factors F, evidence E, and a list of parameters specifying the type of MCMC to be conducted.
	//
	//  G is a data structure that represents the variables in the probabilistic graphical network.  In
	//  particular, it has fields
	//       .names - a list of the names of all variables in the network, in order of variable index
	//       .card - a list of the dimensions of all variables
	//       .var2factors - a mapping of variables to which factors they are included in
	//
	//  F is a list of all factors in the network.
	//  The factor data structure has the following fields:
	//       .var    Vector of variables in the factor, e.g. [1 2 3]
	//       .card    Vector of dimensions corresponding to .var, e.g. [2 2 2]
	//       .val    Value table of size prod(.card)
	//  E is an evidence vector, the same length as G.names, where an entry of 0 means unobserved
	//
	//  TransName is the name of the MCMC transition type (e.g. "Gibbs")
	//
	//  mix_time is the number of iterations to wait until samples are collected.  The user should
	//  determine mix_time by observing behavior using the visualization framework provided.
	//
	//  num_samples is the number of additional samples (after the initial sample
	//  following mixing) to collect
	//
	//  sampling_interval is the number of iterations in the chain to wait between collecting samples
	//  (after mix_time has been reached). This should ALWAYS be set to 1, unless
	//  memory usage is a concern in which case you may want to ignore some samples.
	//
	//  A0 is the initial state of the Markov Chain.  Note that it is a joint assignment to the
	//  variables in G, where element is the value of the variable corresponding to the index.
	//
	//function [M, all_samples] = MCMCInference(G, F, E, TransName, mix_time, ...
	//                                                  num_samples, sampling_interval, A0)
	std::pair< std::vector<Factor>, std::vector<std::vector<uint32_t>> > MCMCInference(
		Graph graph, 
		std::vector<Factor> factors, 
		const Evidence& evidence,
		Trans trans, 
		uint32_t mix_time, 
		uint32_t num_samples,
		uint32_t sampling_interval, 
		std::vector<uint32_t> a0) 
	{
		// observe the evidence
		//for i = 1:length(E),
		//    if (E(i) > 0),
		//        F = ObserveEvidence(F, [i, E(i)]);
		//    end;
		//end;
		ObserveEvidence(factors, evidence);

		// Determine which function to call for Markov Chain transitions
		//bSwendsenWang = false;
		//switch TransName
		// case 'Gibbs'
		//  Trans = @GibbsTrans;
		// case 'MHUniform'
		//  Trans = @MHUniformTrans;
		// case 'MHGibbs'
		//  Trans = @MHGibbsTrans;
		// case 'MHSwendsenWang1'
		//  Trans = @MHSWTrans1;
		//  bSwendsenWang = true;
		// case 'MHSwendsenWang2'
		//  Trans = @MHSWTrans2;
		//  bSwendsenWang = true;
		//end
		//
		//
		// Sampling Loop -----------------------------------
		// Initialize joint assignment
		//A = A0;
		std::vector<uint32_t> a{ a0 };
		//max_iter = mix_time + num_samples * sampling_interval;
		const uint32_t max_iter{ mix_time + num_samples * sampling_interval };
		//all_samples = zeros(max_iter + 1, length(A));
		std::vector<std::vector<uint32_t>> all_samples(max_iter + 1, std::vector<uint32_t>(a.size()));
		//all_samples(1, :) = A0;
		all_samples[0] = a0;
		//disp('Running Markov Chain...');
		std::cout << "Running Markov Chain..." << std::endl;
		std::minstd_rand  gen;
		//for i = 1:max_iter
		for (uint32_t i = 0; i < max_iter; ++i) {
			//    if mod(i, 25) == 0
			if (i  25 == 0) {
				//  disp(['Iteration ', num2str(i)]);
				std::cout << "Iteration " << i << std::endl;
			}//    end
			//    
			//     YOUR CODE HERE
			//     Transition A to the next state in the Markov Chain 
			//     and store the new sample in all_samples
			//    
			//     Note: lines 47-58 use MATLAB's @Function capabilities
			//       this allows binding of function names to a new name
			//       ex:
			//         sol = foo(bar); 
			//         is equivalent to
			//         foo2 = @foo;
			//         sol = foo2(bar);
			//    
			//     This is a dummy line added so that submit.m runs without an error
			//     even if you have not coded anything.
			//     Please delete this line.
			//    all_samples(i+1, :) = A0; 
			// 
			//    A = Trans(A, G, F);
			switch (trans) {
			case Gibbs:
				a = GibbsTrans(a, graph, factors, gen);
				break;
			case MHUniform:
				a = MHUniformTrans(a, graph, factors, gen);
				break;
			case MHGibbs:
				a = MHGibbsTrans(a, graph, factors, gen);
				break;
			case MHSwendsenWang1:
			{
				const auto q_list = CreateQList(graph, factors, UNIFORM);
				a = MHSWTrans(a, graph, factors, gen, UNIFORM, q_list);
				break;
			}
			case MHSwendsenWang2:
			{
				const auto q_list = CreateQList(graph, factors, BLOCK_SAMPLING);
				a = MHSWTrans(a, graph, factors, gen, BLOCK_SAMPLING, q_list);
				break;
			}
			default:
				return {};
			}
			// all_samples(i + 1, :) = A;
			all_samples[i + 1] = a;
		}//end
		//M=[];
		//M = ExtractMarginalsFromSamples(G, all_samples, mix_time+1:sampling_interval:size(all_samples, 1));
		std::vector<uint32_t> collection_indx;
		for (uint32_t i = mix_time + 1; i < all_samples.size(); i += sampling_interval) {
			collection_indx.push_back(i);
		}
		const auto m = ExtractMarginalsFromSamples(graph, all_samples, collection_indx);
		return {m, all_samples};
	}
	//
	//
	//function A = MHSWTrans1(A, G, F)
	//A = MHSWTrans(A, G, F, 1);
	//
	//function A = MHSWTrans2(A, G, F)
	//A = MHSWTrans(A, G, F, 2);

	// MHUNIFORMTRANS
	//
	//  MCMC Metropolis-Hastings transition function that
	//  utilizes the uniform proposal distribution.
	//  A - The current joint assignment.  This should be
	//      updated to be the next assignment
	//  G - The network
	//  F - List of all factors
	//
	//function A = MHUniformTrans(A, G, F)
	std::vector<uint32_t> MHUniformTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand& gen)
	{
		// Draw proposed new state from uniform distribution
		//A_prop = ceil(rand(1, length(A)) .* G.card);
		std::vector<uint32_t> a_prop(assignment.size(),0);
		for (uint32_t i = 0; i < assignment.size(); ++i) 
		{
			std::uniform_int_distribution<int> uni_int_dist(0, graph.card[i] - 1);
			a_prop[i] = uni_int_dist(gen);
		}//

		//p_acceptance = min(1, exp(LogProbOfJointAssignment(F, A_prop) ...
		//                        - LogProbOfJointAssignment(F, A)));
		const auto p_acceptance = std::min(1.0, std::exp(LogProbOfJointAssignment(factors, a_prop) - LogProbOfJointAssignment(factors, assignment)));
	
		std::vector<uint32_t> new_assignment(assignment);
		// Accept or reject proposal
		std::uniform_real_distribution<> uni_real_dist(0.0, 1.0);
		//if rand() < p_acceptance
		if (uni_real_dist(gen) < p_acceptance) 
		{
			//     disp('Accepted');
			//    A = A_prop;
			new_assignment = a_prop;
		}//end
		return new_assignment;
	}

	// MHGIBBSTRANS
	//
	//  MCMC Metropolis-Hastings transition function that
	//  utilizes the Gibbs sampling distribution for proposals.
	//  A - The current joint assignment.  This should be
	//      updated to be the next assignment
	//  G - The network
	//  F - List of all factors
	//
	//function A = MHGibbsTrans(A, G, F)
	std::vector<uint32_t> MHGibbsTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand& gen)
	{
		// Draw proposed new state from Gibbs Transition distribution
		//A_prop = GibbsTrans(A, G, F);
		const auto a_prop = GibbsTrans(assignment, graph, factors, gen);
		//
		// Compute acceptance probability
		//p_acceptance = 1.0;
		double p_acceptance{ 1.0 };

		std::vector<uint32_t> new_assignment(assignment);
		// Accept or reject proposal
		std::uniform_real_distribution<> uni_real_dist(0.0, 1.0);
		//if rand() < p_acceptance
		if (uni_real_dist(gen) < p_acceptance)
		{
			//     disp('Accepted');
			//    A = A_prop;
			new_assignment = a_prop;
		}//end
		return new_assignment;
	}

	// MHSWTRANS
	//
	//  MCMC Metropolis-Hastings transition function that
	//  utilizes the Swendsen-Wang proposal distribution.
	//  A - The current joint assignment.  This should be
	//      updated to be the next assignment
	//  G - The network
	//  F - List of all factors
	//  variant - a number (1 or 2) indicating the variant of Swendsen-Wang to use.  In variant 1,
	//            all the q_{i,j}'s are equal
	//
	//function A = MHSWTrans(A, G, F, variant)
	std::vector<uint32_t> MHSWTrans(const std::vector<uint32_t>& assignment, const Graph& graph, const std::vector<Factor>& factors, std::minstd_rand& gen, SWVariant variant, const QList& q_list)
	{
		// Get Proposal 
		// Prune edges from q_list if the nodes don't have the same current value
		//q_list = G.q_list;
		//q_keep_indx = find(A(q_list(:, 1)) == A(q_list(:, 2)));
		std::vector<QEntry> q_list_keep;
		for (uint32_t i = 0; i < q_list.size(); ++i) {
			if (assignment[q_list[i].node_i] == assignment[q_list[i].node_j])
				q_list_keep.push_back(q_list[i]);
		}
		//q_list = q_list(q_keep_indx, :);
		// Select edges at random based on q_list
		//selected_edges_q_list_indx = find(q_list(:, 3) > rand(size(q_list,1), 1));
		//selected_edges = q_list(selected_edges_q_list_indx, 1:2);
		std::uniform_real_distribution<> uni_real_dist(0.0, 1.0);
		QList selected_edges;
		for (uint32_t i = 0; i < q_list_keep.size(); ++i) {
			if (q_list_keep[i].q_ij > uni_real_dist(gen)) 
			{
				selected_edges.push_back(q_list_keep[i]);
			}
		}
		// Compute connected components over selected edges
		//SelEdgeMat = sparse([selected_edges(:,1)'; selected_edges(:,2)'],...
		//                    [selected_edges(:,2)'; selected_edges(:,1)'],...
		//                    1, length(G.names), length(G.names));
		//
		//[var2comp, cc_sizes] = scomponents(SelEdgeMat);
		const auto cc2var{ FindConnectedComponents(CreateAdjacencyMatrixFromQList(selected_edges, graph.var)) };
		
		//num_cc = length(cc_sizes);
		const auto num_cc{ cc2var.size() };

		// Select a connected component (the book calls this Y)
		//selected_cc = ceil(rand() * num_cc);
		std::uniform_int_distribution<> uni_int_dist(0, num_cc - 1);
		const auto selected_cc = uni_int_dist(gen);
		//selected_vars = find(var2comp == selected_cc);
		const auto selected_vars{ cc2var[selected_cc] };
		
		// Check that the dimensions are all the same and they have the same current assignment
		//assert(length(unique(G.card(selected_vars))) == 1);
		//assert(length(unique(A(selected_vars))) == 1);
		//
		// Pick a new label via sampling
		//old_value = A(selected_vars(1));
		const auto old_value{ assignment[selected_vars[0]] };
		//d = G.card(selected_vars(1));
		const auto d{ graph.card[selected_vars[0]] };
		//LogR = zeros(1, d);
		std::vector<double> log_r(d);
		//if variant == 1
		if (variant == UNIFORM) {
			// logR = ones(1, d) * log(1 / d);
			for (uint32_t i = 0; i < d; ++i) {
				log_r[i] = std::log(1.0 / d);
			}
		}
		//elseif variant == 2
		else if (variant == BLOCK_SAMPLING) {
			// logR = BlockLogDistribution(selected_vars, G, F, A);
			log_r = BlockLogDistribution(selected_vars, graph, factors, assignment);
		} //else
		else 
		{
			// disp('WARNING: Unrecognized Swendsen-Wang Variant');
			std::cout << "WARNING: Unrecognized Swendsen-Wang Variant" << std::endl;
		}//end
		//
		// Sample the new value from the distribution R
		//new_value = randsample(d, 1, true, exp(logR));
		std::vector<double> r(log_r.size());
		std::transform(log_r.begin(), log_r.end(), r.begin(), [](const auto& val) { return std::exp(val); });
		std::vector<uint32_t> vals{ d - 1 };
		const auto new_value = RandSample(vals, 1, true, r, gen)[0];
		//A_prop = A;
		std::vector<uint32_t> a_prop(assignment);
		//A_prop(selected_vars) = new_value;
		for (uint32_t i = 0; i < selected_vars.size(); ++i) {
			a_prop[selected_vars[i]] = new_value;
		}
		//
		// Get the log-ratio of the probability of picking the connected component Y given A_prop over A
		//log_QY_ratio = 0.0;
		double log_QY_ratio{ 0.0 };
		//for i = 1:size(G.q_list, 1)   Iterate through *all* edges, not just the ones we selected earlier
		for (uint32_t i = 0; i < q_list.size(); ++i) {
			//    u = G.q_list(i, 1);
			//    v = G.q_list(i, 2);
			const auto u = q_list[i].node_i;
			const auto v = q_list[i].node_j;
			//    if length(intersect([u, v], selected_vars)) == 1   the edge is from Y to outside-Y
			if (Intersection({ u,v }, selected_vars).values.size() == 1) {
				// if A(u) == old_value && A(v) == old_value
				if (assignment[u] == old_value && assignment[v] == old_value) {
					// log_QY_ratio = log_QY_ratio - log(1 - G.q_list(i, 3));
					log_QY_ratio -= std::log(1.0 - q_list[i].q_ij);
				}// end
				// if A_prop(u) == new_value && A_prop(v) == new_value
				if (a_prop[u] == new_value && a_prop[v] == new_value) {
					// log_QY_ratio = log_QY_ratio + log(1 - G.q_list(i, 3));
					log_QY_ratio += std::log(1.0 - q_list[i].q_ij);
				}// end
			}//    end
		}//end
		//
		//p_acceptance = 0.0;
		double p_acceptance{ 0.0 };
		//p_acceptance = min(1, exp(LogProbOfJointAssignment(F, A_prop) ...
		//                                                   - LogProbOfJointAssignment(F, A) ...
		//                                                   + log_QY_ratio ...
		//                                                   - logR(new_value) ...
		//                                                   + logR(old_value)));
		p_acceptance = std::min(1.0, std::exp(
			LogProbOfJointAssignment(factors, a_prop)
			-LogProbOfJointAssignment(factors, assignment)
			+ log_QY_ratio
			+ log_r[old_value]
			- log_r[new_value]
		));
		
		std::vector<uint32_t> new_assignment(assignment);
		// Accept or reject proposal
		// std::uniform_real_distribution<> uni_real_dist(0.0, 1.0); already defined above
		//if rand() < p_acceptance
		if (uni_real_dist(gen) < p_acceptance)
		{
			//     disp('Accepted');
			//    A = A_prop;
			new_assignment = a_prop;
		}//end
		return new_assignment;
	}

	std::vector <std::vector<uint32_t>> CreateAdjacencyMatrixFromQList(const QList& q_list, const std::vector<uint32_t>& var) 
	{
		std::vector <std::vector<uint32_t>> edges(var.size(), std::vector<uint32_t>(var.size()));
		for (const auto& entry : q_list) {
			edges[entry.node_i][entry.node_j] = 1;
			edges[entry.node_j][entry.node_i] = 1;
		}
		return edges;
	}

	std::vector<std::vector<uint32_t>> FindConnectedComponents(std::vector<std::vector<uint32_t>> edges)
	{
		std::map<uint32_t, bool> all_visited;
		std::vector<std::vector<uint32_t>> cc2var;
		// for all variables
		for (uint32_t i = 0; i < edges.size(); ++i) 
		{
			// if variable not yet visited
			if (!all_visited[i]) 
			{
				// create a new connected component with depth-first-search starting from var
				std::map<uint32_t, bool> visited;
				DFS(i, edges, visited);
				std::vector<uint32_t> vars(visited.size());
				std::transform(visited.begin(), visited.end(), vars.begin(), [](const auto& entry) { return entry.first; });
				cc2var.push_back(vars);
				all_visited.insert(visited.begin(), visited.end());
			}
		}
		return cc2var;
	}

	// Determines all vars that can be reached from var with depth-first-search
	void DFS(const uint32_t var, const std::vector<std::vector<uint32_t>> edges, std::map<uint32_t, bool>& visited) {
		visited[var] = true;
		const std::vector<uint32_t>& edge{ edges[var] };
		for (uint32_t i = 0; i < edge.size(); ++i) 
		{
			if (edge[i] == 1 && !visited[i])
			{
				DFS(i, edges, visited);
			}
		}
	}

}


