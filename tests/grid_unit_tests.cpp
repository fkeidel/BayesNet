#include "gtest/gtest.h"
#include "bayesnet/grid.h"

namespace Bayes
{
	TEST(Grid, Ind2Sub) {
		uint32_t n = 4;
		std::vector<std::pair<uint32_t, uint32_t>> sub_expected{
			{0,0},{0,1},{0,2},{0,3},
			{1,0},{1,1},{1,2},{1,3},
			{2,0},{2,1},{2,2},{2,3},
			{3,0},{3,1},{3,2},{3,3}
		};

		std::vector<std::pair<uint32_t, uint32_t>> sub(n * n);
		for (uint32_t i = 0; i < n * n; ++i) {
			sub[i] = Ind2Sub(n, i);
		}
		EXPECT_EQ(sub, sub_expected);
	}

	TEST(Grid, CreateGridMrf) {
		std::vector<std::vector<uint32_t>> edges_expected =
		{
			{0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
			{1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0},
			{0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0},
			{1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0},
			{0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0},
			{0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0},
			{0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0},
			{0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0},
			{0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0},
			{0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1},
			{0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0},
			{0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1},
			{0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0}
		};

		const auto grid_mrf = CreateGridMrf(4, 0.5, 0.5);
		EXPECT_EQ(grid_mrf.first.edges, edges_expected);
	}

}