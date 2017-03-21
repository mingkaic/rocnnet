//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_node.h"
#include "mocks/mock_connector.h"
#include "fuzz.h"


using namespace testing;


//#define DISABLE_CONNECTOR_TEST
#ifndef DISABLE_CONNECTOR_TEST


// covers iconnector
// copy constructor and assignment
TEST(CONNECTOR, Copy_C000)
{
	mock_connector* assign = new mock_connector({}, "");
	mock_connector* assign2 = new mock_connector({}, "");
	mock_connector* assign3 = new mock_connector({}, "");
	mock_connector* assign4 = new mock_connector({}, "");

	std::string conname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string conname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string label1 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	mock_node* n1 = new mock_node(label1);

	mock_connector* conn = new mock_connector(std::vector<inode<double> *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode<double> *>{n1, n1}, conname2);
	void* ogid1 = conn->get_gid();
	void* ogid2 = conn2->get_gid();

	mock_connector* cpy = new mock_connector(*conn);
	mock_connector* cpy2 = new mock_connector(*conn2);
	void* gid1 = cpy->get_gid();
	void* gid2 = cpy2->get_gid();
	EXPECT_NE(ogid1, gid1);
	EXPECT_NE(ogid2, gid2);

	*assign = *conn;
	*assign2 = *conn2;
	void* gid5 = assign->get_gid();
	void* gid6 = assign2->get_gid();
	EXPECT_NE(ogid1, gid5);
	EXPECT_NE(ogid2, gid6);

	// test base connectors before testing connectors of connectors
	// to prevent optimizations from deleting gids thereby allowing duplicate gid addresses.
	// (without allocator randomization)

	mock_connector* boss = new mock_connector(std::vector<inode<double> *>{conn, n1}, bossname);
	mock_connector* boss2 = new mock_connector(std::vector<inode<double> *>{conn, conn2}, bossname2);
	void* ogid3 = boss->get_gid();
	void* ogid4 = boss2->get_gid();

	mock_connector* cpy3 = new mock_connector(*boss);
	mock_connector* cpy4 = new mock_connector(*boss2);
	void* gid3 = cpy3->get_gid();
	void* gid4 = cpy4->get_gid();
	EXPECT_EQ(ogid3, gid3);
	EXPECT_EQ(ogid4, gid4);

	*assign3 = *boss;
	*assign4 = *boss2;
	void* gid7 = assign3->get_gid();
	void* gid8 = assign4->get_gid();
	EXPECT_EQ(ogid3, gid7);
	EXPECT_EQ(ogid4, gid8);

	EXPECT_CALL(*conn, commit_sudoku()).Times(1);
	EXPECT_CALL(*conn2, commit_sudoku()).Times(2);
	EXPECT_CALL(*boss, commit_sudoku()).Times(2);
	EXPECT_CALL(*boss2, commit_sudoku()).Times(2);
	EXPECT_CALL(*cpy, commit_sudoku()).Times(1);
	EXPECT_CALL(*cpy2, commit_sudoku()).Times(2);
	EXPECT_CALL(*cpy3, commit_sudoku()).Times(2);
	EXPECT_CALL(*cpy4, commit_sudoku()).Times(2);
	EXPECT_CALL(*assign, commit_sudoku()).Times(1);
	EXPECT_CALL(*assign2, commit_sudoku()).Times(2);
	EXPECT_CALL(*assign3, commit_sudoku()).Times(2);
	EXPECT_CALL(*assign4, commit_sudoku()).Times(2);
	delete n1;
	delete conn;
	delete conn2;
	delete boss;
	delete boss2;
	delete assign;
	delete assign2;
	delete assign3;
	delete assign4;
	delete cpy;
	delete cpy2;
	delete cpy3;
	delete cpy4;
}


// covers iconnector
// move constructor and assignment
TEST(CONNECTOR, Move_C000)
{
	mock_connector* assign = new mock_connector({}, "");
	mock_connector* assign2 = new mock_connector({}, "");
	mock_connector* assign3 = new mock_connector({}, "");
	mock_connector* assign4 = new mock_connector({}, "");

	std::string conname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string conname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string label1 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	mock_node* n1 = new mock_node(label1);

	mock_connector* conn = new mock_connector(std::vector<inode<double> *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode<double> *>{n1, n1}, conname2);
	void* ogid1 = conn->get_gid();
	void* ogid2 = conn2->get_gid();

	mock_connector* mv = new mock_connector(std::move(*conn));
	mock_connector* mv2 = new mock_connector(std::move(*conn2));
	EXPECT_EQ(nullptr, conn->get_gid());
	EXPECT_EQ(nullptr, conn2->get_gid());
	EXPECT_EQ(ogid1, mv->get_gid());
	EXPECT_EQ(ogid2, mv2->get_gid());

	*assign = std::move(*mv);
	*assign2 = std::move(*mv2);
	EXPECT_EQ(nullptr, mv->get_gid());
	EXPECT_EQ(nullptr, mv2->get_gid());
	EXPECT_EQ(ogid1, assign->get_gid());
	EXPECT_EQ(ogid2, assign2->get_gid());

	// test base connectors before testing connectors of connectors
	// to prevent optimizations from deleting gids thereby allowing duplicate gid addresses.
	// (without allocator randomization)

	EXPECT_THROW(new mock_connector(std::vector<inode<double> *>{conn, n1}, bossname), std::exception);
	EXPECT_THROW(new mock_connector(std::vector<inode<double> *>{conn, conn2}, bossname2), std::exception);
	mock_connector* boss = new mock_connector(std::vector<inode<double> *>{assign, n1}, bossname);
	mock_connector* boss2 = new mock_connector(std::vector<inode<double> *>{assign, assign2}, bossname2);
	void* ogid3 = boss->get_gid();
	void* ogid4 = boss2->get_gid();

	mock_connector* mv3 = new mock_connector(std::move(*boss));
	mock_connector* mv4 = new mock_connector(std::move(*boss2));
	EXPECT_EQ(nullptr, boss->get_gid());
	EXPECT_EQ(nullptr, boss2->get_gid());
	EXPECT_EQ(ogid3, mv3->get_gid());
	EXPECT_EQ(ogid4, mv4->get_gid());

	*assign3 = std::move(*mv3);
	*assign4 = std::move(*mv4);
	EXPECT_EQ(nullptr, mv3->get_gid());
	EXPECT_EQ(nullptr, mv4->get_gid());
	EXPECT_EQ(ogid3, assign3->get_gid());
	EXPECT_EQ(ogid4, assign4->get_gid());

	EXPECT_CALL(*assign, commit_sudoku()).Times(1);
	EXPECT_CALL(*assign2, commit_sudoku()).Times(2);
	EXPECT_CALL(*assign3, commit_sudoku()).Times(2);
	EXPECT_CALL(*assign4, commit_sudoku()).Times(2);
	delete n1;
	delete conn;
	delete conn2;
	delete boss;
	delete boss2;
	delete assign;
	delete assign2;
	delete assign3;
	delete assign4;
	delete mv;
	delete mv2;
	delete mv3;
	delete mv4;
}


// covers iconnector
// get_name
TEST(CONNECTOR, Name_C001)
{
	size_t nargs = FUZZ<size_t>::get(1, {2, 7})[0];
	std::vector<inode<double>*> ns;
	std::vector<size_t> nlens = FUZZ<size_t>::get(nargs, {14, 29});
	std::string argname = "";
	for (size_t i = 0; i < nargs; i++)
	{
		std::string label = FUZZ_STRING::get(nlens[0]);
		argname += label + ",";
		ns.push_back(new mock_node(label));
	};
	argname.pop_back(); // remove last comma
	std::string bossname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	mock_connector* con1 = new mock_connector(ns, bossname);
	std::string expectname = "<"+bossname+":"+con1->get_uid()+">("+argname+")";
	EXPECT_EQ(expectname, con1->get_name());

	EXPECT_CALL(*con1, commit_sudoku()).Times(nargs);
	for (inode<double>* n : ns)
	{
		delete n;
	}
	delete con1;
}


// covers iconnector
// update_graph, is_same_graph
TEST(CONNECTOR, Graph_C002)
{
	std::string conname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string conname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string label1 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	mock_node* n1 = new mock_node(label1);
	mock_connector* conn = new mock_connector(std::vector<inode<double> *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode<double> *>{n1, n1}, conname2);

	EXPECT_FALSE(conn->is_same_graph(conn2));
	mock_connector* boss = new mock_connector(std::vector<inode<double> *>{conn, n1}, bossname);
	EXPECT_FALSE(conn2->is_same_graph(boss));
	mock_connector* boss2 = new mock_connector(std::vector<inode<double> *>{conn, conn2}, bossname2);

	// boss2 connects conn and conn2
	EXPECT_TRUE(conn->is_same_graph(conn2));
	EXPECT_TRUE(conn2->is_same_graph(boss));

	EXPECT_TRUE(boss->is_same_graph(conn));
	EXPECT_TRUE(boss2->is_same_graph(boss2));

	EXPECT_CALL(*conn, commit_sudoku()).Times(1);
	EXPECT_CALL(*conn2, commit_sudoku()).Times(2);
	EXPECT_CALL(*boss, commit_sudoku()).Times(2);
	EXPECT_CALL(*boss2, commit_sudoku()).Times(2);
	delete n1;
	delete conn;
	delete conn2;
	delete boss;
	delete boss2;
}


// covers iconnector
// potential_descendent
TEST(CONNECTOR, Descendent_C003)
{
	std::string conname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string conname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string bossname2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string label1 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string label2 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	std::string label3 = FUZZ_STRING::get(FUZZ<size_t>::get(1, {14, 29})[0]);
	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_connector* conn = new mock_connector(std::vector<inode<double> *>{n1}, conname);
	mock_connector* conn2 = new mock_connector(std::vector<inode<double> *>{n1, n1}, conname2);
	mock_connector* separate = new mock_connector(std::vector<inode<double>*>{n3, n2}, conname2);
	mock_connector* boss = new mock_connector(std::vector<inode<double> *>{n1, n2}, conname2);

	EXPECT_CALL(*conn, get_leaves(_)).Times(2);
	EXPECT_CALL(*conn2, get_leaves(_)).Times(2);
	EXPECT_CALL(*boss, get_leaves(_)).Times(2);
	EXPECT_CALL(*separate, get_leaves(_)).Times(2);
	conn->potential_descendent(conn2);
	conn2->potential_descendent(conn);
	boss->potential_descendent(separate);
	separate->potential_descendent(boss);

	EXPECT_CALL(*conn, commit_sudoku()).Times(1);
	EXPECT_CALL(*conn2, commit_sudoku()).Times(2);
	EXPECT_CALL(*separate, commit_sudoku()).Times(2);
	EXPECT_CALL(*boss, commit_sudoku()).Times(2);
	delete n1;
	delete n2;
	delete n3;
	delete conn;
	delete conn2;
	delete boss;
	delete separate;
}


#endif /* DISABLE_CONNECTOR_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
