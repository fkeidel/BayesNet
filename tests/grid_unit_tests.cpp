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

	TEST(Grid, Smooth_Given5ValuesAndSpan3) 
	{
		std::vector<uint32_t> y{ 42,7,34,5,9 };
		auto smooth_expected{ y };
		smooth_expected[1] = (y[0] + y[1] + y[2]) / 3;
		smooth_expected[2] = (y[1] + y[2] + y[3]) / 3;
		smooth_expected[3] = (y[2] + y[3] + y[4]) / 3;

		const auto smooth = Smooth(y, 3);

		EXPECT_EQ(smooth, smooth_expected);
	}

	TEST(Grid, Smooth_Given5ValuesAndSpan5)
	{
		//+% !yy2(2) = (y(1) + y(2) + y(3)) / 3;
		//+% !yy2(3) = (y(1) + y(2) + y(3) + y(4) + y(5)) / 5;
		//+% !yy2(4) = (y(3) + y(4) + y(5)) / 3;
		std::vector<uint32_t> y{ 42,7,34,5,9 };
		auto smooth_expected{ y };
		smooth_expected[1] = (y[0] + y[1] + y[2]) / 3;
		smooth_expected[2] = (y[0] + y[1] + y[2] + y[3] + y[4]) / 5;
		smooth_expected[3] = (y[2] + y[3] + y[4]) / 3;

		const auto smooth = Smooth(y, 5);

		EXPECT_EQ(smooth, smooth_expected);
	}


}