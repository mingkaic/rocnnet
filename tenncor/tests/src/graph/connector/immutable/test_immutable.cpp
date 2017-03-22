//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_IMMUTABLE_TEST
#ifndef DISABLE_IMMUTABLE_TEST


TEST(IMMUTABLE, Copy_D000)
{}


TEST(IMMUTABLE, Move_D000)
{}


TEST(IMMUTABLE, Descendent_D001)
{
//	std::string conname = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
//	std::string conname2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
//	std::string bossname = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
//	std::string bossname2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
//	std::string label1 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
//	std::string label2 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
//	std::string label3 = FUZZ::getString(FUZZ::getInt(1, {14, 29})[0]);
//	mock_node* n1 = new mock_node(label1);
//	mock_node* n2 = new mock_node(label2);
//	mock_node* n3 = new mock_node(label3);
//	mock_connector* conn = new mock_connector(std::vector<inode<double> *>{n1}, conname);
//	mock_connector* conn2 = new mock_connector(std::vector<inode<double> *>{n1, n1}, conname2);
//	mock_connector* separate = new mock_connector(std::vector<inode<double>*>{n3, n2}, conname2);
//	mock_connector* boss = new mock_connector(std::vector<inode<double> *>{n1, n2}, conname2);
//
//	EXPECT_TRUE(conn->potential_descendent(conn));
//	EXPECT_TRUE(conn->potential_descendent(conn2));
//	EXPECT_TRUE(conn2->potential_descendent(conn));
//	EXPECT_TRUE(conn2->potential_descendent(conn2));
//	EXPECT_TRUE(boss->potential_descendent(conn));
//	EXPECT_TRUE(boss->potential_descendent(conn2));
//
//	EXPECT_FALSE(separate->potential_descendent(conn));
//	EXPECT_FALSE(separate->potential_descendent(conn2));
//	EXPECT_FALSE(separate->potential_descendent(boss));
//	EXPECT_FALSE(conn->potential_descendent(boss));
//	EXPECT_FALSE(conn2->potential_descendent(boss));
//
//	EXPECT_CALL(*conn, commit_sudoku()).Times(1);
//	EXPECT_CALL(*conn2, commit_sudoku()).Times(2);
//	EXPECT_CALL(*separate, commit_sudoku()).Times(2);
//	EXPECT_CALL(*boss, commit_sudoku()).Times(2);
//	delete n1;
//	delete conn;
//	delete conn2;
//	delete boss;
//	delete separate;
}


TEST(IMMUTABLE, Shape_D002)
{

}


TEST(IMMUTABLE, TensorAndStatus_D003)
{

}


TEST(IMMUTABLE, ImmutableDeath_D004)
{

}


#endif /* DISABLE_IMMUTABLE_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
