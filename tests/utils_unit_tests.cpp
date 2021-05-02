#include <iostream>

#include "gtest/gtest.h"
#include "bayesnet/utils.h"

namespace Bayes {
	TEST(SortIndices, SortIndices)
	{
		std::vector<int> unsorted{ 5,4,3,2,1,0 };
		std::vector<size_t> expected{ 0,1,2,3,4,5 };

		const auto sorted_indices = SortIndices(unsorted);

		for (size_t i = 0; i < unsorted.size(); ++i) {
			EXPECT_EQ(unsorted[sorted_indices[i]], expected[i]);
		}
	}

	TEST(Intersection, Intersection)
	{
		std::vector<int> a{ 0,1,2,3,4,5 };
		std::vector<int> b{ 4, 2 };

		std::vector<int> values_expected{ 2, 4 };
		std::vector<size_t> left_indices_expected{ 2, 4 };
		std::vector<size_t> right_indices_expected{ 1, 0 };

		SetOperationResult<int> result = Intersection(a, b);

		EXPECT_EQ(result.values, values_expected);
		EXPECT_EQ(result.left_indices, left_indices_expected);
		EXPECT_EQ(result.right_indices, right_indices_expected);
	}

	TEST(Union, Union_WhenNotOverlappingAndNoGap)
	{
		std::vector<int> a{ 0,1,2 };
		std::vector<int> b{ 3,4,5 };

		std::vector<int> values_expected{ 0,1,2,3,4,5 };
		std::vector<size_t> left_indices_expected{ 0,1,2 };
		std::vector<size_t> right_indices_expected{ 3,4,5 };

		SetOperationResult<int> result = Union(a, b);

		EXPECT_EQ(result.values, values_expected);
		EXPECT_EQ(result.left_indices, left_indices_expected);
		EXPECT_EQ(result.right_indices, right_indices_expected);
	}

	TEST(Union, Union_WhenOverlapping_ThenOverlappingElementNotDuplicated)
	{
		std::vector<int> a{ 0,1,2 };
		std::vector<int> b{ 2,3,4 };

		std::vector<int> values_expected{ 0,1,2,3,4 };
		std::vector<size_t> left_indices_expected{ 0,1,2 };
		std::vector<size_t> right_indices_expected{ 2,3,4 };

		SetOperationResult<int> result = Union(a, b);

		EXPECT_EQ(result.values, values_expected);
		EXPECT_EQ(result.left_indices, left_indices_expected);
		EXPECT_EQ(result.right_indices, right_indices_expected);
	}

	TEST(Union, Union_WhenGap_ThenGapInUnion)
	{
		std::vector<int> a{ 0,1,2 };
		std::vector<int> b{ 4,5,6 };

		std::vector<int> values_expected{ 0,1,2,4,5,6 };
		std::vector<size_t> left_indices_expected{ 0,1,2 };
		std::vector<size_t> right_indices_expected{ 3,4,5 };

		SetOperationResult<int> result = Union(a, b);

		EXPECT_EQ(result.values, values_expected);
		EXPECT_EQ(result.left_indices, left_indices_expected);
		EXPECT_EQ(result.right_indices, right_indices_expected);
	}

	TEST(Union, Union_Unsorted)
	{
		std::vector<int> a{ 2,1,0 };
		std::vector<int> b{ 6,4,5 };

		std::vector<int> values_expected{ 0,1,2,4,5,6 };
		std::vector<size_t> left_indices_expected{ 2,1,0 };
		std::vector<size_t> right_indices_expected{ 5,3,4 };

		SetOperationResult<int> result = Union(a, b);

		EXPECT_EQ(result.values, values_expected);
		EXPECT_EQ(result.left_indices, left_indices_expected);
		EXPECT_EQ(result.right_indices, right_indices_expected);
	}

	TEST(Difference, Difference)
	{
		std::vector<int> a{ 0,1,2 };
		std::vector<int> b{ 2 };

		std::vector<int> values_expected{ 0,1 };
		std::vector<size_t> left_indices_expected{ 0,1 };

		SetOperationResult<int> result = Difference(a, b);

		EXPECT_EQ(result.values, values_expected);
		EXPECT_EQ(result.left_indices, left_indices_expected);
	}

	TEST(Difference, Difference_WhenUnsorted)
	{
		std::vector<int> a{ 3,2,1 };
		std::vector<int> b{ 3 };

		std::vector<int> values_expected{ 1,2 };
		std::vector<size_t> left_indices_expected{ 2,1 };

		SetOperationResult<int> result = Difference(a, b);

		EXPECT_EQ(result.values, values_expected);
		EXPECT_EQ(result.left_indices, left_indices_expected);
	}




}