#ifndef UTILS_H
#define UTILS_H

#include <vector>
//#include <unordered_set>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <cassert>
#include <set>

namespace Bayes {

	// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
	template <typename T>
	std::vector<size_t> SortIndices(const std::vector<T>& v) {

		// initialize original index locations
		std::vector<size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		std::stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

		return idx;
	}

	template<class T>
	struct SetOperationResult {
		std::vector<T> values;
		std::vector<size_t> left_indices;
		std::vector<size_t> right_indices;
	};

	template <class T>
	SetOperationResult<T> Intersection(std::vector<T> const& left_vector, std::vector<T> const& right_vector) {

		const auto sorted_indices_left = SortIndices(left_vector);
		const auto sorted_indices_right = SortIndices(right_vector);

		auto left_index = sorted_indices_left.begin();
		auto left_index_end = sorted_indices_left.end();
		auto right_index = sorted_indices_right.begin();
		auto right_index_end = sorted_indices_right.end();

		SetOperationResult<T> result;

		while (left_index != left_index_end && right_index != right_index_end) {
			if (left_vector[*left_index] == right_vector[*right_index]) {
				result.values.push_back(left_vector[*left_index]);
				result.left_indices.push_back(*left_index);
				result.right_indices.push_back(*right_index);
				++left_index;
				++right_index;
				continue;
			}

			if (left_vector[*left_index] < right_vector[*right_index]) {
				++left_index;
				continue;
			}

			assert(left_vector[*left_index] > right_vector[*right_index]);
			++right_index;
		}

		return result;
	}

	template <class T>
	SetOperationResult<T> Union(std::vector<T> const& left_vector, std::vector<T> const& right_vector) 
	{
		const auto sorted_indices_left = SortIndices(left_vector);
		const auto sorted_indices_right = SortIndices(right_vector);

		auto left_index = sorted_indices_left.begin();
		auto left_index_end = sorted_indices_left.end();
		auto right_index = sorted_indices_right.begin();
		auto right_index_end = sorted_indices_right.end();

		SetOperationResult<T> result;
		result.left_indices.resize(left_vector.size());
		result.right_indices.resize(right_vector.size());
		size_t result_index = 0;

		while (true)
		{
			if ((left_index == left_index_end) && (right_index == right_index_end)) break;
			else if ((right_index == right_index_end) || ((left_index != left_index_end) && (left_vector[*left_index] < right_vector[*right_index])))
			{ 
				result.values.push_back(left_vector[*left_index]);
				result.left_indices[*left_index] = result_index;
				++left_index;
			}
			else if ((left_index == left_index_end) || ((right_index != right_index_end) && (left_vector[*left_index] > right_vector[*right_index])))
			{ 
				result.values.push_back(right_vector[*right_index]);
				result.right_indices[*right_index] = result_index;
				++right_index;
			}
			else { 
				result.values.push_back(left_vector[*left_index]);
				result.left_indices[*left_index] = result_index;
				++left_index;
				result.right_indices[*right_index] = result_index;
				++right_index;
			}
			++result_index;
		}
		return result;
	}

	template <class T>
	SetOperationResult<T> Difference(std::vector<T> const& left_vector, std::vector<T> const& right_vector) {

		const auto sorted_indices_left = SortIndices(left_vector);
		const auto sorted_indices_right = SortIndices(right_vector);

		auto left_index = sorted_indices_left.begin();
		auto left_index_end = sorted_indices_left.end();
		auto right_index = sorted_indices_right.begin();
		auto right_index_end = sorted_indices_right.end();

		SetOperationResult<T> result;

		while (left_index != left_index_end && right_index != right_index_end)
		{
			if (left_vector[*left_index] < right_vector[*right_index]) { 
				result.values.push_back(left_vector[*left_index]);
				result.left_indices.push_back(*left_index);
				++left_index;

			}
			else if (left_vector[*left_index] > right_vector[*right_index]) { 
				++right_index; 
			}
			else { 
				++left_index; 
				++right_index; 
			}
		}

		while (left_index != left_index_end) {
			result.values.push_back(left_vector[*left_index]);
			result.left_indices.push_back(*left_index);
			++left_index;
		}

		return result;
	}

}
#endif // FACTOR_H