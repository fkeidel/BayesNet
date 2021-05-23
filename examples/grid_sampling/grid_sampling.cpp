#include "bayesnet/clique_tree.h"
#include "examples/example_utils.h"
#include "bayesnet/grid.h"
#include "bayesnet/sampling.h"
#include <chrono>
#include <random>
#include <string>
#include <vector>

using namespace Bayes;

// VISUALIZEMCMCMARGINALS
//
// This function accepts a list of sample lists, each from a different MCMC run.  It then visualizes
// the estimated marginals for each variable in V over the lifetime of the MCMC run.
//
// samples_list - a list of sample lists; each sample list is a m-by-n matrix where m is the
// number of samples and n is the number of variables in the state of the Markov chain
//
// V - an array of variables
// D - the dimensions of the variables in V
// F - a list of factors (used in computing likelihoods of each sample)
// window_size - size of the window over which to aggregate samples to compute the estimated
//               marginal at a given time
// ExactMarginals - the exact marginals of V (optional)
//
//function VisualizeMCMCMarginalsFile(plotsdir, samples_list, V, D, F, window_size, ExactMarginals, tname)
void VisualizeMCMCMarginals();
//function VisualizeMCMCMarginalsFile(plotsdir, samples_list, V, D, F, window_size, ExactMarginals, tname)
//
//for i = 1:length(V)
//    figure('visible', 'off')
//    v = V(i);
//    d = D(i);
//    title(['Marginal for Variable ', num2str(v)]);
//    if exist('ExactMarginals') == 1, M = ExactMarginals(i); end;
//    for j = 1:length(samples_list)
//        samples_v = samples_list{j}(:, v);
//        indicators_over_time = zeros(length(samples_v), d);
//        for k = 1:length(samples_v)
//            indicators_over_time(k, samples_v(k)) = 1;
//        end
//
//        % estimated_marginal = cumsum(indicators_over_time, 1);
//        estimated_marginal = [];
//        for k = 1:size(indicators_over_time, 2)
//            estimated_marginal = [estimated_marginal, smooth(indicators_over_time(:, k), window_size)];
//        end
//        % Prune ends
//        estimated_marginal = estimated_marginal(window_size/2:end - window_size/2, :);
//
//
//        estimated_marginal = estimated_marginal ./ ...
//            repmat(sum(estimated_marginal, 2), 1, size(estimated_marginal, 2));
//        hold on;
//        plot(estimated_marginal, '-', 'LineWidth', 2);
//        title(['Est marginals for entry ' num2str(i) ' of samples for ' tname])
//        if exist('M') == 1
//            plot([1; size(estimated_marginal, 1)], [M.val; M.val], '--', 'LineWidth', 3);
//        end
//        set(gcf,'DefaultAxesColorOrder', rand(d, 3));
//    end
//    print('-dpng', [plotsdir, '/MCMC_', tname, '.png']);
//end
//
// Visualize likelihood of sample at each time step
//all_likelihoods = [];
//for i = 1:length(samples_list)
//    samples = samples_list{i};
//    likelihoods = [];
//    for j = 1:size(samples, 1)
//        likelihoods = [likelihoods; LogProbOfJointAssignment(F, samples(j, :))];
//    end
//    all_likelihoods = [all_likelihoods, likelihoods];
//end
//figure('visible', 'off')
//plot(all_likelihoods, '-', 'LineWidth', 2);
//title(['Likelihoods for ' tname])
//print('-dpng', [plotsdir, '/LIKE_', tname, '.png']);


//function VisualizeToyImageMarginals(G, M, chain_num, tname)
void VisualizeToyImageMarginals(Graph graph, std::vector<Factor> marginals, uint32_t chain_num, std::string trans_name, std::string file_path)
{
	//n = sqrt(length(G.names));
	const auto n{ std::sqrt(graph.var.size()) };
	//marginal_vector = [];
	std::vector<std::vector<double>> marginal_matrix(n, std::vector<double>(n));
	//for i = 1:length(M)
	for (uint32_t i = 0; i < marginals.size(); ++i)
	{
		//    marginal_vector(end+1) = M(i).val(1);
		const auto sub{ Ind2Sub(n, i) };
		const auto row{ sub.first };
		const auto col{ sub.second };
		marginal_matrix[row][col] = marginals[i].Val(0); // or 0?
	}//end
	//clims = [0, 1];
	//imagesc(reshape(marginal_vector, n, n), clims);
	//colormap(gray);
	//title(['Marginals for chain ' num2str(chain_num) ' ' tname])
	std::string title{ "Marginals for chain " + std::to_string(chain_num) + " " + trans_name };
	WriteTableToCsv(file_path + trans_name + ".csv", title, marginal_matrix);
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
	uint32_t num_chains_to_run{ 3 };
	uint32_t mix_time{ 400 };
	uint32_t collect{ 6000 };
	double on_diagonal{ 0.3 };
	double off_diagonal{ 1 };
		
	// Directory to save the plots into, change to your output path
	//plotsdir = './plots_test';
	const std::string file_path{ "c:\\BayesNet\\examples\\grid_sampling\\plots\\" };
	//
	//start = time;
	auto start = std::chrono::system_clock::now();
		
	// Construct grid markov random field
	//[toy_network, toy_factors] = ConstructToyNetwork(on_diagonal, off_diagonal);
	auto grid_mrf = CreateGridMrf(4, on_diagonal, off_diagonal);
	const auto& graph{ grid_mrf.first };
	auto& factors{ grid_mrf.second };
	// 
	//toy_evidence = zeros(1, length(toy_network.names));
	//%toy_clique_tree = CreateCliqueTree(toy_factors, []);
	//%toy_cluster_graph = CreateClusterGraph(toy_factors,[]);
	//
	// Exact Inference
	//ExactM = ComputeExactMarginalsBP(toy_factors, toy_evidence, 0);
	const Evidence NO_EVIDENCE;
	const auto exact_marginals = CliqueTreeComputeExactMarginalsBP(factors, NO_EVIDENCE, false);

	//graphics_toolkit('gnuplot');
	//figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, ExactM, 1, 'Exact');
	VisualizeToyImageMarginals(graph, exact_marginals, 1, "Exact", file_path);
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
	//errors = {};
	//
	//total_cycles = length(transition_names) * num_chains_to_run;
	//cycles_so_far = 0;
	//for j = 1:length(transition_names)
	//    samples_list = {};
	//    errors_list = [];
	//
	//    for i = 1:num_chains_to_run
	//        % Random Initialization
	//        A0 = ceil(rand(1, length(toy_network.names)) .* toy_network.card);
	//
	//        % Initialization to all ones
	//        % A0 = i * ones(1, length(toy_network.names));
	//
	//        MCMCstart = time;
	//        [M, all_samples] = ...
	//            MCMCInference(toy_network, toy_factors, toy_evidence, transition_names{j}, mix_time, collect, 1, A0);
	//        samples_list{i} = all_samples;
	//        disp(['MCMCInference took ', num2str(time-MCMCstart), ' sec.']);
	//        fflush(stdout);
	//        errors_list(:, i) = CalculateErrors(toy_network, ExactM, all_samples, mix_time);
	//	err_start = time;
	//	disp(['Calculating errors took: ', num2str(time-err_start), ' sec.']);
	//	fflush(stdout);
	//
	//        figure('visible', 'off'), VisualizeToyImageMarginals(toy_network, M, i, transition_names{j}); 
	//        print('-dpng', [plotsdir, '/GREY_', transition_names{j}, '_sample', num2str(i), '.png']);
	//
	//        cycles_so_far = cycles_so_far + 1;
	//        cycles_left =  (total_cycles - cycles_so_far);
	//	timeleft = ((time - start) / cycles_so_far) * cycles_left;
	//        disp(['Progress: ', num2str(cycles_so_far), '/', num2str(total_cycles), ...
	//              ', estimated time left to complete: ', num2str(timeleft), ' sec.']);
	//  
	//    end
	//    errors{j} = errors_list;
	//
	//    vis_vars = [3];
	//    VisualizeMCMCMarginalsFile(plotsdir, samples_list, vis_vars, toy_network.card(vis_vars), toy_factors, ...
	//      500, ExactM(vis_vars),transition_names{j});
	//    VisualizeConvergence(plotsdir, samples_list, [3 10], ExactM([3 10]), transition_names{j});
	//
	//    disp(['Saved results for MCMC with transition ', transition_names{j}]);
	//end
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