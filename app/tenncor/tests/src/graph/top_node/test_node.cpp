//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef DISABLE_TOP_NODE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "mocks/mock_node.h"
#include "fuzz.h"


#ifndef DISABLE_NODE_TEST


// covers inode
// copy assignment and constructor
TEST(NODE, Copy_B000)
{
	FUZZ::reset_logger();
	mock_node assign;

	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	mock_node n1(label1);

	mock_node cpy(n1);
	assign = n1;

	EXPECT_EQ(label1, n1.get_label());
	EXPECT_EQ(n1.get_label(), cpy.get_label());
	EXPECT_EQ(n1.get_label(), assign.get_label());

	EXPECT_NE(n1.get_uid(), cpy.get_uid());
	EXPECT_NE(n1.get_uid(), assign.get_uid());
}


// covers inode
// move assignment and constructor
TEST(NODE, Move_B000)
{
	FUZZ::reset_logger();
	mock_node assign;

	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	mock_node n1(label1);

	std::string ouid = n1.get_uid();
	EXPECT_EQ(label1, n1.get_label());

	mock_node mv(std::move(n1));

	std::string ouid2 = mv.get_uid();
	EXPECT_TRUE(n1.get_label().empty());
	EXPECT_EQ(label1, mv.get_label());
	EXPECT_NE(ouid, ouid2);

	assign = std::move(mv);

	std::string ouid3 = assign.get_uid();
	EXPECT_TRUE(mv.get_label().empty());
	EXPECT_EQ(label1, assign.get_label());
	EXPECT_NE(ouid, ouid3);
	EXPECT_NE(ouid2, ouid3);
}


// covers inode
// get_uid
TEST(NODE, UID_B001)
{
	FUZZ::reset_logger();
	std::unordered_set<std::string> us;
	size_t ns = FUZZ::getInt(1, "ns", {1412, 2922})[0];
	for (size_t i = 0; i < ns; i++)
	{
		mock_node mn;
		std::string uid = mn.get_uid();
		EXPECT_TRUE(us.end() == us.find(uid));
		us.emplace(uid);
	}
}


// covers inode
// get_label, get_name
TEST(NODE, Label_B002)
{
	FUZZ::reset_logger();
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	mock_node n1(label1);

	std::string uid = n1.get_uid();
	EXPECT_EQ(n1.get_name(), "<"+label1+":"+uid+">");
	EXPECT_EQ(label1, n1.get_label());
}


#endif /* DISABLE_NODE_TEST */


#endif /* DISABLE_TOP_NODE_MODULE_TESTS */
