
#include <thread>

#include "gtest/gtest.h"

#include "fmts/fmts.hpp"

#include "perf/measure.hpp"

int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}


static void mock_measure (perf::PerfRecord& record)
{
	perf::MeasureScope _defer(&record, "f1");

	std::this_thread::sleep_for(
		std::chrono::milliseconds(1000));
}


TEST(PERFORMANCE, Measure)
{
	perf::PerfRecord record;
	mock_measure(record);

	std::stringstream ss;
	record.to_csv(ss);
	auto got = ss.str();
	auto lines = fmts::split(got, "\n");
	EXPECT_EQ(3, lines.size());
	EXPECT_EQ(0, lines[2].size());
	ASSERT_NE(0, lines[1].size());
	ASSERT_NE(0, lines[0].size());

	auto parts = fmts::split(lines[1], ",");
	ASSERT_EQ(4, parts.size());
	EXPECT_STREQ("f1", parts[0].c_str());
	EXPECT_STREQ("1", parts[3].c_str());
	EXPECT_STREQ(parts[1].c_str(), parts[2].c_str());
	EXPECT_EQ(10, parts[1].size()); // expect duration to be in the billions
	EXPECT_STREQ("function,mean duration(ns),total duration(ns),n occurrences", lines[0].c_str());
}
