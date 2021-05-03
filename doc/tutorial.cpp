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
   Factor rain{ {RAIN}, {2}, { 0.86, 0.14 } };
   Factor rain_given_season{ {RAIN,SEASON}, {2,2}, {0.82,0.18,0.86,0.14} };

   // a factor is a function object
   // for an assignment to the parent variables, it returns a real value
   const auto value = rain_given_season({FALSE,TRUE});
   std::cout << "get value of assignement:\nvalue = " << value << std::endl;

   // factor product
   Factor season{ {SEASON}, {2}, { 0.5, 0.5 } };
   Factor joint = FactorProduct(season, rain_given_season);
   std::cout << "\nfactor product:\njoint:\n" << joint;

    // factor marginalization
    // remove Season from joint
    Factor marginal = joint.Marginalize({SEASON});
    std::cout << "marginalization\nmarginal:\n" << marginal;


}