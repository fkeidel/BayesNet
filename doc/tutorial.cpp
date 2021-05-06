#include "bayesnet/factor.h"
#include "bayesnet/clique_tree.h"
#include <iostream>
#include <iomanip>
#include <map>

using namespace Bayes;

int main()
{
   std::cout << "Tutorial\n\n" << std::endl;

   // factors
   std::cout << "******************" << std::endl;
   std::cout << "Factor calculation" << std::endl;
   std::cout << "******************" << std::endl;
   // random variables have unique, contiguous its
   enum VarIds : uint32_t {
      SEASON,
      RAIN
   };

   // values of the discete variables
   enum RainValues : uint32_t {
      NO,
      YES
   };

   enum SeasnonValues : uint32_t {
      WINTER,
      SUMMER
   };

   // Instantiations of factors (variables of the factor, cardinalities, values)
   const Factor pd_rain{ {RAIN}, {2}, { 0.86, 0.14 } };
   Factor cpd_rain_given_season{ {RAIN,SEASON}, {2,2}, {0.82,0.18,0.86,0.14} };

   // a factor is a function object
   // for an assignment to the parent variables, it returns a real value
   const auto value = cpd_rain_given_season({NO,SUMMER});
   std::cout << "get value of assignement:\nvalue = " << value << std::endl;

   // factor product
   const Factor pd_season{ {SEASON}, {2}, { 0.5, 0.5 } };
   const Factor joint_season_rain = FactorProduct(pd_season, cpd_rain_given_season);
   std::cout << "\nfactor product:\njoint_season_rain:\n" << joint_season_rain;

   // factor marginalization
   // remove Season from joint
   Factor m_rain = joint_season_rain.Marginalize({SEASON});
   std::cout << "marginalization\nm_rain:\n" << m_rain;

   // observe evidence
   const Evidence evidence{ {SEASON, SUMMER} };
   cpd_rain_given_season.ObserveEvidence(evidence);
   std::cout << "observe evidence\ncpd_rain_given_season(Season=true):\n" << cpd_rain_given_season;

   bool marginalize{ true };
   cpd_rain_given_season.ObserveEvidence(evidence, marginalize);
   std::cout << "observe evidence (marginalize)\ncpd_rain_given_season(Season=true):\n" << cpd_rain_given_season;

   // let's add another variable country
   // add additional id 
   const uint32_t COUNTRY{ 2 };
   // country has two values {false, true} which mean {Finland, Egypt}
   // the numbers in the factor represent the number of inhabitants in millions
   // if the factor should represent a probability, we have to normalize the factor
   Factor pd_country{ {COUNTRY}, {2}, { 5, 100 } };
   pd_country.Normalize();
   // now change the factors for Season and Rain to have an additional dependency to country
   Factor cpd_season_given_country{ {SEASON,COUNTRY}, {2,2}, {0.8,0.2,0.2,0.8} };
   Factor cpd_rain_given_country_and_season{ {RAIN,SEASON,COUNTRY}, {2,2,2}, {0.82,0.18,0.86,0.14,0.4,0.6,0.6,0.4} };

   // inference algorithms
   std::cout << "********************" << std::endl;
   std::cout << "inference algorithms" << std::endl;
   std::cout << "********************" << std::endl;

   std::map<uint32_t, std::string> labels
   {
      { SEASON, "Season" },
      { RAIN, "Rain" },
      { COUNTRY, "Country" }
   };

   std::vector <Factor> factors{
      pd_country,
      cpd_season_given_country,
      cpd_rain_given_country_and_season
   };

   // Variable Elimination
   std::cout << "\nVariable Elimination" << std::endl;
   auto f(factors);  // make copy to preserve original factors
   // eliminate Season and Country gives the marginal of Rain
   VariableElimination(f, { SEASON,COUNTRY });
   m_rain = f.front();
   std::cout << std::fixed << std::setprecision(2) << labels[m_rain.Var(0)] << ":\n" << m_rain;

   f = factors; // restore original factors
   // eliminate Rain and Country gives the marginal of Season
   VariableElimination(f, { RAIN,COUNTRY });
   const auto m_season = f.front();
   std::cout << std::fixed << std::setprecision(2)  << labels[m_season.Var(0)] << ":\n" << m_season;

   // Clique Tree
   std::cout << "\nClique Tree" << std::endl;
   f = factors;  // make copy to preserve original factors
   // Step 1: Creation
   CliqueTree c{ f, {} };
   // Step 2: Calibration
   c.Calibrate();
   // Step 3: Retrieval
   // get ordered list of all variables
   std::vector<uint32_t> vars{ UniqueVars(c.CliqueList()) };
   // calculate marginals for all variables
   std::vector<Factor> m(vars.size());
   // for all variables and cliques
   for (const auto& var : vars) {
      for (const auto& clique : c.CliqueList()) {
         // if variable is in scope of clique
         if (!Intersection({ var }, clique.Var()).values.empty()) {
            // marginalize all variables except current variable to get marginal        
            m[var] = clique.Marginalize(Difference(clique.Var(), { var }).values);
            m[var].Normalize();
         }
      }
   }
   // print results
   for (const auto marginal : m) {
      std::cout << std::fixed << std::setprecision(2) << labels[marginal.Var(0)] << ":\n" << marginal;
   }

   // You can do all 3 steps in 1
   f = factors;  // make copy to preserve original factors
   m = CliqueTreeComputeExactMarginalsBP(f, {}, false);






}