#include "factor.h"
#include <iostream>
#include <iomanip>

// Disease-Test is an example of the application of Bayes Rule
// cf.
// [1] http://www.openmarkov.org/docs/tutorial/ - chapter 1
// and
// [2] https://machinelearningmastery.com/bayes-theorem-for-machine-learning/


using namespace Bayes;

int main()
{
	std::cout << "Disease-Test\n\n";

	enum DiseaseTestVars : uint32_t {
		DISEASE,
		TEST
	};

	enum DiscreteValues : uint32_t {
		FALSE,
		TRUE
	};

	// Values from [1]
	double prevalence { 0.14 };
	double sensitivity{ 0.9  }; // true positive rate (TPR)
	double specificity{ 0.93 }; // true negative rate (TNR)
	
	// create factores of (conditional) probability distributions
	Factor p_disease{            {DISEASE},      {2},   {1- prevalence,prevalence} };
	Factor p_test_given_disease{ {TEST,DISEASE}, {2,2}, {specificity, 1-specificity, 1-sensitivity,sensitivity} };
	std::vector<Factor> factors{ p_disease, p_test_given_disease };

	// evidence
	Evidence e_test_true{ {TEST , TRUE} };
	
	auto f{ factors }; // fresh factors
	// Simple calculation of P(Disease|Test=true) 
	// Calculates the joint probability of all factors and then marginalizes.
	// Do not use SimpleComputeMarginal for large networks.
	// For large networks use VariableElimination, see below
	auto m_disease = SimpleComputeMarginal({ DISEASE }, f, e_test_true);

	std::cout << "1. Values from http://www.openmarkov.org/docs/tutorial/\n\n"
		<< std::fixed << std::setprecision(2)
		<< "P(Disease=true)             = " << prevalence << " [prevalence]\n"
		<< "P(Test=true|Disease=true)   = " << sensitivity << " [sensitivity]\n"
		<< "P(Test=false|Disease=false) = " << specificity << " [specificity]\n\n"
		<< "1.1 manual calculation\n\n"
		<< "----------------------\n"
		<< "|P(Disease|Test=true)|\n"
		<< "|--------------------|\n"
		<< "|   FALSE  |  TRUE   |\n"
		<< "|----------|---------|\n"
		<< "|   " << m_disease({ FALSE }) << "   |  " << m_disease({ TRUE }) << "   |\n"
		<< "----------------------\n\n";


	// Use variable elimination
	// In this case, the network is small and we could use also SimpleComputeMarginal, see above.
	// But we use VariableElimination here, to show how it is used.
	f = factors; // fresh factors
	// observe evidence
	ObserveEvidence(f, e_test_true);
	// eliminate Test
	VariableElimination(f, { TEST });
	// the result of variable elimination is a list of unnormalized factors, so we have to compute 
	// the joint distribution and normalize to get a marginal probability
	m_disease = ComputeJointDistribution(f);
	m_disease.Normalize();

	std::cout << "1.2 variable elimination\n\n"
		<< "----------------------\n"
		<< "|P(Disease|Test=true)|\n"
		<< "|--------------------|\n"
		<< "|   FALSE  |  TRUE   |\n"
		<< "|----------|---------|\n"
		<< "|   " << m_disease({ FALSE }) << "   |  " << m_disease({ TRUE }) << "   |\n"
		<< "----------------------\n\n";

	// Values from [2]
	prevalence = 0.0002;
	sensitivity = { 0.85 }; // true positive rate (TPR)
	specificity = 0.95 ; // true negative rate (TNR)
	p_disease = { {DISEASE}, {2}, {1 - prevalence,prevalence} };
	p_test_given_disease = { {TEST,DISEASE}, {2,2}, {specificity, 1-specificity, 1-sensitivity,sensitivity} };
	factors = { p_disease, p_test_given_disease };

	ObserveEvidence(factors, e_test_true);
	VariableElimination(factors, { TEST });
	m_disease = ComputeJointDistribution(factors);
	m_disease.Normalize();

	std::cout << "2. Values from https://machinelearningmastery.com/bayes-theorem-for-machine-learning/\n\n"
		<< std::fixed << std::setprecision(4)
		<< "P(Disease=true)             = " << prevalence << " [prevalence]\n"
		<< "P(Test=true|Disease=true)   = " << sensitivity << " [sensitivity]\n"
		<< "P(Test=false|Disease=false) = " << specificity << " [specificity]\n\n"
		<< "----------------------\n"
		<< "|P(Disease|Test=true)|\n"
		<< "|--------------------|\n"
		<< "|   FALSE  |  TRUE   |\n"
		<< "|----------|---------|\n"
		<< "|  " << m_disease({ FALSE }) << "  | " << m_disease({ TRUE }) << "  |\n"
		<< "----------------------\n";

	return 0;
}
