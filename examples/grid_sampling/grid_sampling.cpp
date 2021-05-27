#include "bayesnet/clique_tree.h"
#include "examples/example_utils.h"
#include "bayesnet/grid.h"
#include "bayesnet/sampling.h"
#include <chrono>
#include <random>
#include <string>
#include <vector>

using namespace Bayes;

//% VISUALIZEMCMCMARGINALS
//%
//% This function accepts a list of sample lists, each from a different MCMC run.  It then visualizes
//% the estimated marginals for each variable in V over the lifetime of the MCMC run.
//%
//% samples_list - a list of sample lists; each sample list is a m-by-n matrix where m is the
//% number of samples and n is the number of variables in the state of the Markov chain
//%
//% V - an array of variables
//% D - the dimensions of the variables in V
//% F - a list of factors (used in computing likelihoods of each sample)
//% window_size - size of the window over which to aggregate samples to compute the estimated
//%               marginal at a given time
//% ExactMarginals - the exact marginals of V (optional)
//%
//% Copyright (C) Daphne Koller, Stanford University, 2012
//
//function VisualizeMCMCMarginals(samples_list, V, D, F, window_size, ExactMarginals, tname)
void VisualizeMCMCMarginals(std::vector<std::vector<std::vector<uint32_t>>> samples_list,
	uint32_t v, uint32_t d, std::vector<Factor> factors, uint32_t window_size, Factor exact_marginal, std::string trans_name, std::string file_path
)
{
	//for i = 1:length(V)
	//    figure;
	//    v = V(i);
	//    d = D(i);
	//    title(['Marginal for Variable ', num2str(v)]);
	//    if exist('ExactMarginals') == 1, M = ExactMarginals(i); end;
	//    for j = 1:length(samples_list)
	for (uint32_t j = 0; j < samples_list.size(); ++j)
	{
		const auto& samples{ samples_list[j] };
		// samples_v = samples_list{j}(:, v);
		std::vector<uint32_t> samples_v(samples.size());
		for (uint32_t k = 0; k < samples.size(); ++k) {
			samples_v[k] = samples[k][v];
		}
		// indicators_over_time = zeros(length(samples_v), d);
		std::vector<std::vector<double>> indicators_over_time(d, std::vector<double>(samples_v.size(), 0.0));
		// for k = 1:length(samples_v)
		for (uint32_t k = 0; k < samples_v.size(); ++k)
		{
			//     indicators_over_time(k, samples_v(k)) = 1;
			indicators_over_time[samples_v[k]][k] = 1.0;
		}// end
		//
		// % estimated_marginal = cumsum(indicators_over_time, 1);
		// estimated_marginal = [];
		std::vector<std::vector<double>> estimated_marginal(d);
		// for k = 1:size(indicators_over_time, 2)
		for (uint32_t k = 0; k < d; ++k)
		{		// 
		//     estimated_marginal = [estimated_marginal, smooth(indicators_over_time(:, k), window_size)];
			estimated_marginal[k] = Smooth(indicators_over_time[k], window_size);
		}// end
		// % Prune ends
		// estimated_marginal = estimated_marginal(window_size/2:end - window_size/2, :);
		const auto half_window{ (window_size - 1) / 2 };
		for (uint32_t k = 0; k < d; ++k)
		{		
			estimated_marginal[k].erase(estimated_marginal[k].begin(), estimated_marginal[k].begin() + half_window);
			estimated_marginal[k].erase(estimated_marginal[k].end() - half_window, estimated_marginal[k].end());
		}
		//
		// estimated_marginal = estimated_marginal ./ ...
		//     repmat(sum(estimated_marginal, 2), 1, size(estimated_marginal, 2));
		// hold on;
		// plot(estimated_marginal, '-', 'LineWidth', 2);
		// title(['Est marginals for entry ' num2str(i) ' of samples for ' tname])
		const std::string header{"false,true"};
		WriteTableToCsv(file_path + "Chart_Est_Marginals_Var" + std::to_string(v) + "_" + trans_name + "_" + std::to_string(j) + ".csv",
			estimated_marginal, header, true);
		// if exist('M') == 1
		//     plot([1; size(estimated_marginal, 1)], [M.val; M.val], '--', 'LineWidth', 3);
		// end
		// set(gcf,'DefaultAxesColorOrder', rand(d, 3));
	}//    end
	//end
	//
	// Visualize likelihood of sample at each time step
	//all_likelihoods = [];
	//for i = 1:length(samples_list)
	std::vector<std::vector<double>> all_likelihoods(samples_list.size());
	std::string header;
	for (uint32_t i = 0; i < samples_list.size(); ++i)
	{
		// samples = samples_list{i};
		const auto& samples{ samples_list[i] };
		// likelihoods = [];
		std::vector<double> likelihoods(samples.size());
		// for j = 1:size(samples, 1)
		for (uint32_t j = 0; j < samples.size(); ++j)
		{
			//     likelihoods = [likelihoods; LogProbOfJointAssignment(F, samples(j, :))];
			likelihoods[j] = LogProbOfJointAssignment(factors, samples[j]);
		}// end
		// all_likelihoods = [all_likelihoods, likelihoods];
		all_likelihoods[i] = likelihoods;
		header.append("run " + std::to_string(i));
		if (i < samples_list.size()-1) 
		{
			header.append(",");
		}
	} //end
	//figure;
	//title('Likelihoods')
	WriteTableToCsv(file_path + "Likelihoods_" + trans_name +  ".csv", all_likelihoods, header, true);
	//plot(all_likelihoods, '-', 'LineWidth', 2);
}

//function VisualizeToyImageMarginals(G, M, chain_num, tname)
void VisualizeImageMarginals(Graph graph, std::vector<Factor> marginals, uint32_t chain_num, std::string trans_name, std::string file_path)
{
	//n = sqrt(length(G.names));
	const auto n{ std::sqrt(graph.var.size()) };
	//marginal_vector = [];
	std::vector<std::vector<double>> marginal_matrix(n, std::vector<double>(n));
	//for i = 1:length(M)
	for (uint32_t i = 0; i < marginals.size(); ++i)
	{
		// marginal_vector(end+1) = M(i).val(1);
		const auto sub{ Ind2Sub(n, i) };
		const auto row{ sub.first };
		const auto col{ sub.second };
		marginal_matrix[row][col] = marginals[i].Val(1); 
	}//end
	//clims = [0, 1];
	//imagesc(reshape(marginal_vector, n, n), clims);
	//colormap(gray);
	//title(['Marginals for chain ' num2str(chain_num) ' ' tname])
	WriteTableToCsv(file_path + "Image_" + trans_name + "_" + std::to_string(chain_num) + ".csv", marginal_matrix);
}

//  linear absolute error
//  m = [M(:).val];
//  er = sum(abs([ExactM(:).val] - m)) / length(m);
double GetError(Factor exact_m, Factor m)
{
	std::vector<double> abs_errors(m.Val().size());
	for (uint32_t i = 0; i < m.Val().size(); ++i) {
		abs_errors[i] = std::abs(exact_m.Val(i) - m.Val(i));
	}
	return std::accumulate(abs_errors.begin(), abs_errors.end(), 0.0);
}

//function er = GetError3(ExactM, M)
std::vector<double> GetErrors (std::vector<Factor> exact_marginals, std::vector<Factor> marginals) 
{
	std::vector<double> errors(marginals.size());
	for (uint32_t i = 0; i < errors.size(); ++i) {
		errors[i] = GetError(exact_marginals[i], marginals[i]);
	}
	return errors;
}//end


	//function errors = CalculateErrors(G, TrueMargs, all_samples, mix_time)
	//% Calculates error distance between all true marginals and samples marginals at each step
	//% Sampled marginals after mix_time will be calculated without the data before mix_time
std::vector<double> CalculateErrors(Graph g, std::vector<Factor> true_m, std::vector<std::vector<uint32_t>> all_samples, uint32_t mix_time) 
{
	//
	//  GetError = @GetError3;  % select which error metric should be used
	//  [nsamples, nvars] = size(all_samples);
	const auto n_samples = all_samples.size();
	//
	//%  all_m = {};
	//  
	//  EmptyM = repmat(struct('var', 0, 'card', 0, 'val', []), length(G.names), 1);
	//  for i = 1:length(G.names)
	//    EmptyM(i).var = i;
	//    EmptyM(i).card = G.card(i);
	//    EmptyM(i).val = zeros(1, G.card(i));
	//  end
	//  M = EmptyM;
	//  PrevM = EmptyM;
	//
	//  for s = [1:nsamples]
	for (uint32_t s = mix_time; s < n_samples; ++s)
	{
		//    sample = all_samples(s,:);
		//    if (mix_time == s)      % reset the counters
		//      PrevM = EmptyM;
		//    end
		//    NormM = EmptyM;
		//    for j=1:length(sample)
		//      PrevM(j).val(sample(j)) = PrevM(j).val(sample(j)) + 1;
		//    end
		//    NormM = PrevM;
		//    for j= [1:length(sample)]
		//      NormM(j).val = NormM(j).val ./ s;
		//    end
		//%    all_m{s} = M;    % no need to store marginals on each step at this moment
		//    errors(s) = GetError(TrueMargs, NormM);
	}//  end
	//end
	//
	//% Error metric functions, possibly not quite adequate? L.F.
	//% change to whatever score measurement technique you like.
	//
	//
	//function er = GetError1(ExactM, M)
	//  er = norm([ExactM(:).val] - [M(:).val]) / norm([ExactM(:).val] + [M(:).val]);
	//end
	//
	//function er = GetError2(ExactM, M)
	//  % play with setting the input mean (0.5), truncate all results above 1
	//  exact = [ExactM(:).val] .- 0.5;
	//  m = [M(:).val] .- 0.5;
	//  er = min([1 abs(norm(exact - m) / norm(exact + m))]);
	//end
	//
	//function er = GetError3(ExactM, M)
	//  % linear absolute error
	//  m = [M(:).val];
	//  er = sum(abs([ExactM(:).val] - m)) / length(m);
	//end
	return{};
}


//  Based on scripts by Binesh Bannerjee (saving to file)
//  and Christian Tott (VisualizeConvergence).
//  Additionally plots a chart with (possibly inadequate) error score
//  relative to all exact marginals
//
//  These scripts depend on PA4 files for exact inference. Either
//  copy them to current dir, or add:
//  addpath '/path/to/your/PA4/Files'
//
//  If you notice that your MCMC wanders in circles, it may be
//  because rand function included with the assignment is still buggy
//  in your course run. In this case rename rand.m and randi.m
//  to some other names (like rand.bk and randi.bk). Don't forget to
//  rename them back if you are going to run test or submit scripts
//  that depend on them.
//
int main()
{
	std::cout << "Grid Sampling (Markov Random Field)" << std::endl;
	//rand('seed', 1);
	std::minstd_rand gen;

	// Tunable parameters
	//num_chains_to_run = 3;
	//mix_time = 400;
	//collect = 6000;
	//on_diagonal = 1;
	//off_diagonal = 0.2;
	// 
	const uint32_t grid_size{ 4U };
	double on_diagonal{ 1.0 };
	double off_diagonal{ 0.2 };

	uint32_t num_chains_to_run{ 3 };
	uint32_t mix_time{ 0 };
	uint32_t collect{ 1600 };
		
	// Directory to save the plots into, change to your output path
	//plotsdir = './plots_test';
	const std::string file_path{ "c:\\BayesNet\\examples\\grid_sampling\\plots\\" };
	//
	//start = time;
	const auto start = std::chrono::system_clock::now();
		
	// Construct grid markov random field
	//[toy_network, toy_factors] = ConstructToyNetwork(on_diagonal, off_diagonal);

	auto grid_mrf = CreateGridMrf(grid_size, on_diagonal, off_diagonal);
	const auto& graph{ grid_mrf.first };
	auto& factors{ grid_mrf.second };

	const auto n{ std::sqrt(graph.var.size()) };
	std::vector<std::vector<double>> singleton_vals(n, std::vector<double>(n));
	for (uint32_t i = 0; i < graph.var.size(); ++i) {
		// marginal_vector(end+1) = M(i).val(1);
		const auto sub{ Ind2Sub(n, i) };
		const auto row{ sub.first };
		const auto col{ sub.second };
		singleton_vals[row][col] = factors[i].Val(1);
	}
	WriteTableToCsv(file_path + "SingletonFactors.csv", singleton_vals);

	// 
	//toy_evidence = zeros(1, length(toy_network.names));
	//%toy_clique_tree = CreateCliqueTree(toy_factors, []);
	//%toy_cluster_graph = CreateClusterGraph(toy_factors,[]);
	//
	// Exact Inference
	//ExactM = ComputeExactMarginalsBP(toy_factors, toy_evidence, 0);
	std::cout << "Run exact inference" << std::endl;
	const Evidence NO_EVIDENCE;
	const auto exact_marginals = CliqueTreeComputeExactMarginalsBP(factors, NO_EVIDENCE, false);

	//graphics_toolkit('gnuplot');
	//figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, ExactM, 1, 'Exact');
	VisualizeImageMarginals(graph, exact_marginals, 0, "Exact", file_path);
	//print('-dpng', [plotsdir, '/EXACT.png']);
	//
	// Comment this in to run Approximate Inference on the toy network
	// Approximate Inference
	// % ApproxM = ApproxInference(toy_cluster_graph, toy_factors, toy_evidence);
	//% figure, VisualizeToyImageMarginals(toy_network, ApproxM);
	// ^^ boobytrap, don't uncomment
	//
	// MCMC Inference
	//transition_names = {'Gibbs', 'MHUniform', 'MHGibbs', 'MHSwendsenWang1', 'MHSwendsenWang2'};
	std::vector<Trans> trans
	{  
		//Gibbs,
		//MHUniform,
		//MHGibbs,
		//MHSwendsenWang1,
		MHSwendsenWang2
	};

	std::vector<std::string> trans_names
	{
		//"Gibbs",
		//"MHUniform",
		//"MHGibbs",
		//"MHSwendsenWang1",
		"MHSwendsenWang2"
	};

	//errors = {};
	//
	//total_cycles = length(transition_names) * num_chains_to_run;
	//uint32_t total_cycles{ trans.size() * num_chains_to_run };
	//cycles_so_far = 0;
	//uint32_t cycles_so_far{ 0U };
	//for j = 1:length(transition_names)
	for (uint32_t j = 0; j < trans.size(); ++j) {
		std::cout << "Running " << trans_names[j] << std::endl;
		//    samples_list = {};
		std::vector<std::vector<std::vector<uint32_t>>> samples_list(num_chains_to_run);
		//    errors_list = [];
		//
		// for i = 1:num_chains_to_run
		for (uint32_t i = 0; i < num_chains_to_run; ++i)
		{
			std::cout << "... chain " << i << std::endl;
			// Random Initialization
			// A0 = ceil(rand(1, length(toy_network.names)) .* toy_network.card);
			std::vector<uint32_t> a0(graph.var.size(), 0);
			for (uint32_t i = 0; i < graph.var.size(); ++i)
			{
				std::uniform_int_distribution<int> uni_int_dist(0, graph.card[i] - 1);
				a0[i] = uni_int_dist(gen);
			}//
			// % Initialization to all ones
			// % A0 = i * ones(1, length(toy_network.names));
			//
			// MCMCstart = time;
			const auto mcmc_start = std::chrono::system_clock::now();
			// [M, all_samples] = ...
			//     MCMCInference(toy_network, toy_factors, toy_evidence, transition_names{j}, mix_time, collect, 1, A0);
			const auto result = MCMCInference(graph, factors, NO_EVIDENCE, trans[j], mix_time, collect, 1, a0);
			const auto& marginals{ result.first };
			const auto& all_samples{ result.second };

			// samples_list{i} = all_samples;
			samples_list[i] = all_samples;
			// disp(['MCMCInference took ', num2str(time-MCMCstart), ' sec.']);
			std::cout << "MCMCInference took " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - mcmc_start).count() << " sec" << std::endl;
			// fflush(stdout);
			// errors_list(:, i) = CalculateErrors(toy_network, ExactM, all_samples, mix_time);
			//	err_start = time;
			//	disp(['Calculating errors took: ', num2str(time-err_start), ' sec.']);
			//	fflush(stdout);
			//
			// figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, M, i, transition_names{j}); 
			VisualizeImageMarginals(graph, marginals, i, trans_names[j], file_path);
			// print('-dpng', [plotsdir, '/GREY_', transition_names{j}, '_sample', num2str(i), '.png']);
			//
			// cycles_so_far = cycles_so_far + 1;
			// cycles_left =  (total_cycles - cycles_so_far);
			//	timeleft = ((time - start) / cycles_so_far) * cycles_left;
			// disp(['Progress: ', num2str(cycles_so_far), '/', num2str(total_cycles), ...
			//       ', estimated time left to complete: ', num2str(timeleft), ' sec.']);
			//  
		}//    end
		//    errors{j} = errors_list;
		//
		//    vis_vars = [3];
		//    VisualizeMCMCMarginalsFile(plotsdir, samples_list, vis_vars, toy_network.card(vis_vars), toy_factors, ...
		//      500, ExactM(vis_vars),transition_names{j});
		uint32_t vis_var{ 2 };
		VisualizeMCMCMarginals(samples_list, vis_var, graph.card[vis_var], factors, 501, exact_marginals[vis_var], trans_names[j], file_path);
		// 
		//    VisualizeConvergence(plotsdir, samples_list, [3 10], ExactM([3 10]), transition_names{j});
		//
		//    disp(['Saved results for MCMC with transition ', transition_names{j}]);
	}//end
	//
	//VisualizeErrors(plotsdir, errors, mix_time, transition_names);
	//
	//elapsed = time - start;
	//
	//fname = [plotsdir, '/report.txt'];
	//file = fopen(fname, 'a');
	//fdisp(file, ['On diag: ', num2str(on_diagonal),
	//             'Off diag: ', num2str(off_diagonal),
	//             'Mix time: ', num2str(mix_time),
	//             'Collect: ', num2str(collect),
	//             'Time consumed: ', num2str(elapsed), ' sec.']);
	//fclose(file);
	//
	//disp(['Done, time consumed: ', num2str(elapsed), ' sec.']);
	return 0;	
}