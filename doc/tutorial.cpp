#include "bayesnet/factor.h"
#include <iostream>

using namespace Bayes;

int main()
{
   std::cout << "Tutorial\n\n";

   // factors

   // random variables have unique, contiguous its
   enum VarIds : uint32_t {
      SEASON,
      RAIN
   };

   // values of a binary random variable
   enum BinaryValue : uint32_t {
      FALSE,
      TRUE
   };

   // Instantiations of factors (variables of the factor, cardinalities, values)
   const Factor rain{ {RAIN}, {2}, { 0.86, 0.14 } };
   Factor rain_given_season{ {RAIN,SEASON}, {2,2}, {0.82,0.18,0.86,0.14} };

   // a factor is a function object
   // for an assignment to the parent variables, it returns a real value
   const auto value = rain_given_season({FALSE,TRUE});
   std::cout << "get value of assignement:\nvalue = " << value << std::endl;

   // factor product
   const Factor season{ {SEASON}, {2}, { 0.5, 0.5 } };
   const Factor joint = FactorProduct(season, rain_given_season);
   std::cout << "\nfactor product:\njoint:\n" << joint;

   // factor marginalization
   // remove Season from joint
   const Factor marginal = joint.Marginalize({SEASON});
   std::cout << "marginalization\nmarginal:\n" << marginal;

   // observe evidence
   const Evidence evidence{ {SEASON, TRUE} };
   rain_given_season.ObserveEvidence(evidence);
   std::cout << "observe evidence\nrain_given_season(Season=true):\n" << rain_given_season;

   bool marginalize{ true };
   rain_given_season.ObserveEvidence(evidence, marginalize);
   std::cout << "observe evidence (marginalize)\nrain_given_season(Season=true):\n" << rain_given_season;

   // let's add another variable country
   // add additional id 
   const uint32_t COUNTRY{ 2 };
   // country has two values {false, true} which mean {Finland, Egypt}
   // the numbers in the factor represent the number of inhabitants in millions
   // if the factor should represent a probability, we have to normalize the factor
   Factor country{ {COUNTRY}, {2}, { 5, 100 } };
   country.Normalize();
   // now change the factors for Season and Rain to have an additional dependency to country
   Factor season_given_country{ {SEASON,COUNTRY}, {2,2}, {0.8,0.2,0.2,0.8} };
   Factor rain_given_country_and_season{ {RAIN,SEASON,COUNTRY}, {2,2,2}, {0.82,0.18,0.86,0.14,0.4,0.6,0.6,0.4} };

   // inference algorithms
   std::cout << "inference algorithms" << std::endl;
   std::vector <Factor> factors{
      country,
      season_given_country,
      rain_given_country_and_season
   };

   // Variable Elimination
   std::cout << "\nVariable Elimination" << std::endl;
   auto f(factors);  // make copy to preserve original factors
   // eliminate Season and Country gives the marginal of Rain
   VariableElimination(f, { SEASON,COUNTRY });
   const auto m_rain = f.front();
   std::cout << "m_rain:\n" << m_rain;

   f = factors; // restore original factors
   // eliminate Rain and Country gives the marginal of Season
   VariableElimination(f, { RAIN,COUNTRY });
   const auto m_season = f.front();
   std::cout << "m_season:\n" << m_season;

}