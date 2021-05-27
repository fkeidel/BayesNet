// based on Coursera course 'Probabilistic Graphical Models' by Daphne Koller, Stanford University
// see https://www.coursera.org/specializations/probabilistic-graphical-models

#ifndef GRID_H
#define GRID_H

#include "bayesnet/factor.h"
#include "bayesnet/sampling.h"
#include <cassert>

namespace Bayes 
{
	std::pair<uint32_t, uint32_t> Ind2Sub(uint32_t cols, uint32_t i);
	std::pair<Graph, std::vector<Factor>> CreateGridMrf(uint32_t n, double weight_of_agreement, double weight_of_disagreement);

	template<class T>
	std::vector<T> Smooth(std::vector<T> y, uint32_t span) 
	{
		// ## Perform smoothing
		// 
		// if (span > length (y))
		//   error ('smooth: span cannot be greater than ''length (y)''.')
		// endif
		assert(("smooth: span cannot be greater than y.size()", span <= y.size()));
		assert(("smooth: span must be odd", (span % 2) != 0));
		// yy = [];
		std::vector<T> avg(y.size());
		// for i=1:length (y)
		for (uint32_t i = 0; i < y.size(); ++i) 
		{
			uint32_t idx1{ 0U };
			uint32_t idx2{ 0U };
			// if (mod (span,2) == 0)
			//   error ('smooth: span must be odd.')
			// endif
			// if (i <= (span-1)/2)
			if (i < (span - 1) / 2) 
			{
				// ## We're in the beginning of the vector, use as many y values as 
				// ## possible and still having the index i in the center.
				// ## Use 2*i-1 as the span.
				// idx1 = 1;
				// idx2 = 2*i-1;
				idx1 = 0;
				idx2 = 2*i;
			}// elseif (i <= length (y) - (span-1)/2)
			else if (i < y.size() - (span-1)/2 ) 
			{
				// ## We're somewhere in the middle of the vector.
				// ## Use full span.
				// idx1 = i-(span-1)/2;
				// idx2 = i+(span-1)/2;
				idx1 = i - (span - 1) / 2;
				idx2 = i + (span - 1) / 2;
			}// else
			else
			{
				// ## We're near the end of the vector, reduce span.
				// ## Use 2*(length (y) - i) + 1 as span
				// idx1 = i - (length (y) - i);
				// idx2 = i + (length (y) - i);
				idx1 = i - (y.size() - 1 - i);
				idx2 = i + (y.size() - 1 - i);
			}//  endif
			// yy(i) = mean (y(idx1:idx2));
			T sum{};
			for (uint32_t j = idx1; j <= idx2; ++j) 
			{
				sum += y[j];
			}
			avg[i] = sum / (idx2 - idx1 + 1);
		}//     endfor
		return avg;
	}
}

#endif // GRID_H