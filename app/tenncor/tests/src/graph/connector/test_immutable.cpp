//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_CONNECTOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "mocks/mock_immutable.h"
#include "mocks/mock_node.h"
#include "mocks/mock_tensor.h"
#include "graph/leaf/variable.hpp"


#ifndef DISABLE_IMMUTABLE_TEST


#ifdef SLOW_GRAPH
static std::pair<size_t,size_t> nnodes_range = {131, 297};
#elif defined(THOROUGH_GRAPH) 
static std::pair<size_t,size_t> nnodes_range = {67, 131};
#elif defined(FAST_GRAPH)
static std::pair<size_t,size_t> nnodes_range = {31, 67};
#else // FASTEST_GRAPH
static std::pair<size_t,size_t> nnodes_range = {17, 31};
#endif


static bool bottom_up (std::vector<iconnector<double>*> ordering)
{
	// ordering travels from leaf towards the root
	// ordering test ensures get leaf is a bottom-up procedure
	// todo: some how test caching performance (probabilistically increase hit rate as i increases)
	// eventually most nodes in traversals should be cached, so ordering size should decrease
	bool o = true;
	iconnector<double>* last = nullptr;
	for (iconnector<double>* ord : ordering)
	{
		if (last)
		{
			// ord should be parent of last
			o = o && ord->has_subject(last);
		}
		last = ord;
	}
	return o;
}


TEST(IMMUTABLE, Copy_I000)
{
	FUZZ::reset_logger();
	immutable<double>* assign  = new mock_immutable(std::vector<inode<double>*>{}, "");
	immutable<double>* central = new mock_immutable(std::vector<inode<double>*>{}, "");
	const tensor<double>* res = central->eval();

	immutable<double>* cpy = central->clone();
	*assign = *central;
	ASSERT_NE(nullptr, cpy);

	const tensor<double>* cres = cpy->eval();
	const tensor<double>* ares = assign->eval();

	std::vector<double> data = expose(central);
	std::vector<double> cdata = expose(cpy);
	std::vector<double> adata = expose(assign);

	EXPECT_TRUE(tensorshape_equal(res->get_shape(), cres->get_shape()));
	EXPECT_TRUE(tensorshape_equal(res->get_shape(), ares->get_shape()));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), cdata.begin()));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), adata.begin()));

	delete assign;
	delete cpy;
	delete central;
}


TEST(IMMUTABLE, Move_I000)
{
	FUZZ::reset_logger();
	immutable<double>* assign  = new mock_immutable(std::vector<inode<double>*>{}, "");
	immutable<double>* central = new mock_immutable(std::vector<inode<double>*>{}, "");
	const tensor<double>* res = central->eval();
	std::vector<double> data = expose(central);
	tensorshape rs = res->get_shape();

	immutable<double>* mv = central->move();
	EXPECT_NE(nullptr, mv);

	const tensor<double>* mres = mv->eval();
	std::vector<double> mdata = expose(mv);
	tensorshape ms = mres->get_shape();

	EXPECT_EQ(nullptr, central->eval());

	*assign = std::move(*mv);
	const tensor<double>* ares = assign->eval();
	std::vector<double> adata = expose(assign);
	tensorshape as = ares->get_shape();

	EXPECT_EQ(nullptr, mv->eval());

	EXPECT_TRUE(tensorshape_equal(rs, ms));
	EXPECT_TRUE(tensorshape_equal(rs, as));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), mdata.begin()));
	EXPECT_TRUE(std::equal(data.begin(), data.end(), adata.begin()));

	delete assign;
	delete mv;
	delete central;
}


TEST(IMMUTABLE, Descendent_I001)
{
	FUZZ::reset_logger();
	std::string conname = FUZZ::getString(FUZZ::getInt(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = FUZZ::getString(FUZZ::getInt(1, "conname2.size", {14, 29})[0], "conname2");
	std::string bossname = FUZZ::getString(FUZZ::getInt(1, "bossname.size", {14, 29})[0], "bossname");
	std::string bossname2 = FUZZ::getString(FUZZ::getInt(1, "bossname2.size", {14, 29})[0], "bossname2");
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = FUZZ::getString(FUZZ::getInt(1, "label3.size", {14, 29})[0], "label3");
	std::vector<double> leafvalue = FUZZ::getDouble(3, "leafvalue");
	variable<double>* n1 = new variable<double>(leafvalue[0], label1);
	variable<double>* n2 = new variable<double>(leafvalue[1], label2);
	variable<double>* n3 = new variable<double>(leafvalue[2], label3);

	immutable<double>* conn = new mock_immutable(std::vector<inode<double> *>{n1}, conname);
	immutable<double>* conn2 = new mock_immutable(std::vector<inode<double> *>{n1, n1}, conname2);
	immutable<double>* separate = new mock_immutable(std::vector<inode<double>*>{n3, n2}, conname2);
	immutable<double>* boss = new mock_immutable(std::vector<inode<double> *>{n1, n2}, conname2);

	EXPECT_TRUE(conn->potential_descendent(conn));
	EXPECT_TRUE(conn->potential_descendent(conn2));
	EXPECT_TRUE(conn2->potential_descendent(conn));
	EXPECT_TRUE(conn2->potential_descendent(conn2));
	EXPECT_TRUE(boss->potential_descendent(conn));
	EXPECT_TRUE(boss->potential_descendent(conn2));

	EXPECT_FALSE(separate->potential_descendent(conn));
	EXPECT_FALSE(separate->potential_descendent(conn2));
	EXPECT_FALSE(separate->potential_descendent(boss));
	EXPECT_FALSE(conn->potential_descendent(boss));
	EXPECT_FALSE(conn2->potential_descendent(boss));

	delete conn;
	delete conn2;
	delete boss;
	delete separate;

	delete n1;
	delete n2;
	delete n3;
}


TEST(IMMUTABLE, Status_I002)
{
	FUZZ::reset_logger();
	std::string conname = FUZZ::getString(FUZZ::getInt(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = FUZZ::getString(FUZZ::getInt(1, "conname2.size", {14, 29})[0], "conname2");
	std::string conname3 = FUZZ::getString(FUZZ::getInt(1, "conname3.size", {14, 29})[0], "conname3");
	std::string conname4 = FUZZ::getString(FUZZ::getInt(1, "conname4.size", {14, 29})[0], "conname4");
	std::string conname5 = FUZZ::getString(FUZZ::getInt(1, "conname5.size", {14, 29})[0], "conname5");
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = FUZZ::getString(FUZZ::getInt(1, "label3.size", {14, 29})[0], "label3");
	std::string label4 = FUZZ::getString(FUZZ::getInt(1, "label4.size", {14, 29})[0], "label4");

	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_node* n4 = new mock_node(label4);

	tensorshape n1s = random_def_shape();
	tensorshape n2s = random_def_shape();
	tensorshape n3s = random_def_shape();
	n1->data_ = new mock_tensor(n1s);
	n2->data_ = new mock_tensor(n2s);
	n3->data_ = new mock_tensor(n3s);

	immutable<double>* conn = new mock_immutable({n1}, conname);
	immutable<double>* conn2 = new mock_immutable({n2, n3}, conname2);
	// bad statuses
	immutable<double>* conn3 = new mock_immutable({n4, n3}, conname3);
	immutable<double>* conn4 = new mock_immutable({n1, n4}, conname4);
	immutable<double>* conn5 = new mock_immutable({n2, n4}, conname5);

	EXPECT_TRUE(conn->good_status());
	EXPECT_FALSE(conn2->good_status());
	conn2->eval();
	EXPECT_TRUE(conn2->good_status());
	EXPECT_FALSE(conn3->good_status());
	EXPECT_FALSE(conn4->good_status());
	EXPECT_FALSE(conn5->good_status());

	delete conn;
	delete conn2;
	delete conn3;
	delete conn4;
	delete conn5;
	delete n1;
	delete n2;
	delete n3;
	delete n4;
}


TEST(IMMUTABLE, Shape_I003)
{
	FUZZ::reset_logger();
	std::string conname = FUZZ::getString(FUZZ::getInt(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = FUZZ::getString(FUZZ::getInt(1, "conname2.size", {14, 29})[0], "conname2");
	std::string conname3 = FUZZ::getString(FUZZ::getInt(1, "conname3.size", {14, 29})[0], "conname3");
	std::string conname4 = FUZZ::getString(FUZZ::getInt(1, "conname4.size", {14, 29})[0], "conname4");
	std::string conname5 = FUZZ::getString(FUZZ::getInt(1, "conname5.size", {14, 29})[0], "conname5");
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = FUZZ::getString(FUZZ::getInt(1, "label3.size", {14, 29})[0], "label3");
	std::string label4 = FUZZ::getString(FUZZ::getInt(1, "label4.size", {14, 29})[0], "label4");

	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_node* n4 = new mock_node(label4); // status is bad

	// mock tensors initialize with random data...
	tensorshape n1s = random_def_shape(2, 10, 17, 4372);
	tensorshape n2s = random_def_shape(2, 10, 17, 4372);
	tensorshape n3s = random_def_shape(2, 10, 17, 4372);
	n1->data_ = new mock_tensor(n1s);
	n2->data_ = new mock_tensor(n2s);
	n3->data_ = new mock_tensor(n3s);

	// for this test, we only care about shape
	auto fittershaper = [](std::vector<tensorshape> ts) -> tensorshape
	{
		std::vector<size_t> res;
		for (tensorshape& s : ts)
		{
			std::vector<size_t> slist = s.as_list();
			size_t minrank = std::min(slist.size(), res.size());
			for (size_t i = 0; i < minrank; i++)
			{
				res[i] = std::max(res[i], slist[i]);
			}
			if (slist.size() > res.size())
			{
				for (size_t i = minrank; i < slist.size(); i++)
				{
					res.push_back(slist[i]);
				}
			}
		}
		return res;
	};

	immutable<double>* conn = new mock_immutable({n1}, conname, fittershaper);
	immutable<double>* conn2 = new mock_immutable({n2, n3}, conname2, fittershaper);
	// bad statuses
	immutable<double>* conn3 = new mock_immutable({n4, n3}, conname3, fittershaper);
	immutable<double>* conn4 = new mock_immutable({n1, n4}, conname4, fittershaper);
	immutable<double>* conn5 = new mock_immutable({n2, n4}, conname5, fittershaper);

	// sample expectations
	tensorshape c2shape = fittershaper({n2s, n3s});

	EXPECT_TRUE(tensorshape_equal(n1s, conn->get_shape()));
	EXPECT_TRUE(tensorshape_equal(c2shape, conn2->get_shape()));

	// bad status returns undefined shapes
	EXPECT_FALSE(conn3->get_shape().is_part_defined()); // not part defined is undefined
	EXPECT_FALSE(conn4->get_shape().is_part_defined());
	EXPECT_FALSE(conn5->get_shape().is_part_defined());

	// delete connectors before nodes to avoid triggering suicides
	delete conn;
	delete conn2;
	delete conn3;
	delete conn4;
	delete conn5;
	delete n1;
	delete n2;
	delete n3;
	delete n4;
}


TEST(IMMUTABLE, Tensor_I004)
{
	FUZZ::reset_logger();
	std::string conname = FUZZ::getString(FUZZ::getInt(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = FUZZ::getString(FUZZ::getInt(1, "conname2.size", {14, 29})[0], "conname2");
	std::string conname3 = FUZZ::getString(FUZZ::getInt(1, "conname3.size", {14, 29})[0], "conname3");
	std::string conname4 = FUZZ::getString(FUZZ::getInt(1, "conname4.size", {14, 29})[0], "conname4");
	std::string conname5 = FUZZ::getString(FUZZ::getInt(1, "conname5.size", {14, 29})[0], "conname5");
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = FUZZ::getString(FUZZ::getInt(1, "label2.size", {14, 29})[0], "label2");
	std::string label3 = FUZZ::getString(FUZZ::getInt(1, "label3.size", {14, 29})[0], "label3");
	std::string label4 = FUZZ::getString(FUZZ::getInt(1, "label4.size", {14, 29})[0], "label4");

	mock_node* n1 = new mock_node(label1);
	mock_node* n2 = new mock_node(label2);
	mock_node* n3 = new mock_node(label3);
	mock_node* n4 = new mock_node(label4); // status is bad

	tensorshape n1s = random_def_shape();
	tensorshape n2s = random_def_shape();
	tensorshape n3s = random_def_shape();
	n1->data_ = new mock_tensor(n1s);
	n2->data_ = new mock_tensor(n2s);
	n3->data_ = new mock_tensor(n3s);

	// for this test, we care about data, grab the largest shape, and sum all data that fit in said array
	auto minshaper = [](std::vector<tensorshape> ts)
	{
		tensorshape res = ts[0];
		for (size_t i = 1, n = ts.size(); i < n; i++)
		{
			if (res.n_elems() > ts[i].n_elems())
			{
				res = ts[i];
			}
		}
		return res;
	};

	immutable<double>* conn = new mock_immutable(
		{n1}, conname, minshaper, adder);
	immutable<double>* conn2 = new mock_immutable(
		{n2, n3}, conname, minshaper, adder);
	// bad statuses
	immutable<double>* conn3 = new mock_immutable(
		{n4, n3}, conname3, minshaper, adder);
	immutable<double>* conn4 = new mock_immutable(
		{n1, n4}, conname4, minshaper, adder);
	immutable<double>* conn5 = new mock_immutable(
		{n2, n4}, conname5, minshaper, adder);

	tensorshape t2 = n2->get_shape();
	tensorshape t3 = n3->get_shape();
	tensorshape c2s = minshaper({t2, t3});
	size_t nc2s = c2s.n_elems();
	std::vector<double> vn2 = expose(n2);
	std::vector<double> vn3 = expose(n3);
	ASSERT_EQ(nc2s, std::min(vn2.size(), vn3.size()));
	ASSERT_EQ(vn2.size(), t2.n_elems());
	ASSERT_EQ(vn3.size(), t3.n_elems());
	double* expectc2 = new double[nc2s];
	{
		std::vector<double> expectin;
		size_t n = std::min(vn2.size(), vn3.size());
		for (size_t i = 0; i < n; i++)
		{
			expectin.push_back(vn2[i]);
			expectin.push_back(vn3[i]);
		}
		for (size_t i = n; i < vn2.size(); i++)
		{
			expectin.push_back(vn2[i]);
			expectin.push_back(0);
		}
		for (size_t i = n; i < vn3.size(); i++)
		{
			expectin.push_back(0);
			expectin.push_back(vn3[i]);
		}

		std::vector<double> v2 = expose<double>(n2);
		std::vector<double> v3 = expose<double>(n3);
		std::vector<const double*> vsinput = {&v2[0], &v3[0]};
		adder(expectc2, vsinput, shape_io{minshaper({n2s, n3s}), {n2s, n3s} });
	}

	const tensor<double>* c1tensor = conn->eval();
	const tensor<double>* c2tensor = conn2->eval();
	ASSERT_NE(nullptr, c1tensor);
	ASSERT_NE(nullptr, c2tensor);

	std::vector<double> n1out = expose(n1);
	std::vector<double> c1out = c1tensor->expose();
	std::vector<double> c2out = c2tensor->expose();
	EXPECT_TRUE(std::equal(n1out.begin(), n1out.end(), c1out.begin()));
	EXPECT_TRUE(std::equal(expectc2, expectc2 + nc2s, c2out.begin()));
	// bad status returns undefined shapes
	EXPECT_EQ(nullptr, conn3->eval()); // not part defined is undefined
	EXPECT_EQ(nullptr, conn4->eval());
	EXPECT_EQ(nullptr, conn5->eval());

	delete[] expectc2;
	delete conn;
	delete conn2;
	delete conn3;
	delete conn4;
	delete conn5;
	delete n1;
	delete n2;
	delete n3;
	delete n4;
}


TEST(IMMUTABLE, ImmutableDeath_I005)
{
	FUZZ::reset_logger();
	size_t nnodes = FUZZ::getInt(1, "nnodes", nnodes_range)[0];
	std::unordered_set<immutable<double>*> leaves;
	std::unordered_set<immutable<double>*> collector;

	// build a tree out of mock immutables
	FUZZ::buildNTree<immutable<double> >(2, nnodes,
	[&leaves](void)
	{
		std::string llabel = FUZZ::getString(FUZZ::getInt(1, "llabel.size", {14, 29})[0], "llabel");
		immutable<double>* im = new mock_immutable(std::vector<inode<double>*>{}, llabel);
		leaves.emplace(im);
		return im;
	},
	[&collector](std::vector<immutable<double>*> args)
	{
		std::string nlabel = FUZZ::getString(FUZZ::getInt(1, "nlabel.size", {14, 29})[0], "nlabel");
		mock_immutable* im = new mock_immutable(
			std::vector<inode<double>*>(args.begin(), args.end()), nlabel);
		im->triggerOnDeath =
		[&collector](mock_immutable* ded)
		{
			collector.erase(ded);
		};
		collector.insert(im);
		return im;
	});

	// check if collectors are all dead
	for (immutable<double>* l : leaves)
	{
		delete l;
	}

	EXPECT_TRUE(collector.empty());
	for (immutable<double>* im : collector)
	{
		delete im;
	}
}


TEST(IMMUTABLE, TemporaryEval_I006)
{
	FUZZ::reset_logger();
	size_t nnodes = FUZZ::getInt(1, "nnodes", nnodes_range)[0];

	std::unordered_set<inode<double>*> leaves;
	std::unordered_set<immutable<double>*> collector;

	tensorshape shape = random_def_shape();
	double single_rando = FUZZ::getDouble(1, "single_rando", {1.1, 2.2})[0];

	auto unifiedshaper =
	[&shape](std::vector<tensorshape>)
	{
		return shape;
	};

	const_init<double> cinit(single_rando);

	inode<double>* root = FUZZ::buildNTree<inode<double> >(2, nnodes,
	[&leaves, &shape, &cinit]() -> inode<double>*
	{
		std::string llabel = FUZZ::getString(FUZZ::getInt(1, "llabel.size", {14, 29})[0], "llabel");
		variable<double>* im = new variable<double>(shape, cinit, llabel);
		im->initialize();
		leaves.emplace(im);
		return im;
	},
	[&collector, &unifiedshaper](std::vector<inode<double>*> args)
	{
		std::string nlabel = FUZZ::getString(FUZZ::getInt(1, "nlabel.size", {14, 29})[0], "nlabel");
		mock_immutable* im = new mock_immutable(args, nlabel, unifiedshaper, adder);
		im->triggerOnDeath =
		[&collector](mock_immutable* ded)
		{
			collector.erase(ded);
		};
		collector.insert(im);
		return im;
	});

	inode<double>* out = nullptr;
	std::unordered_set<ileaf<double>*> lcache;
	for (immutable<double>* coll : collector)
	{
		if (coll == root) continue;
		lcache.clear();
		static_cast<immutable<double>*>(root)->temporary_eval(coll, out);
		ASSERT_NE(nullptr, out);
		const tensor<double>* outt = out->eval();
		ASSERT_NE(nullptr, outt);
		ASSERT_TRUE(tensorshape_equal(shape, outt->get_shape()));
		// out data should be 1 + M * single_rando where M is the
		// number of root's leaves that are not in coll's leaves
		lcache = coll->get_leaves();
		size_t M = leaves.size() - lcache.size();
		double datum = M * single_rando + 1;
		std::vector<double> odata = outt->expose();
		double diff = std::abs(datum - odata[0]);
		EXPECT_GT(0.000001 * single_rando, diff); // allow error of a tiny fraction of the random leaf value
		delete out;
		out = nullptr;
	}

	for (inode<double>* l : leaves)
	{
		delete l;
	}
	for (immutable<double>* im : collector)
	{
		delete im;
	}
}


TEST(IMMUTABLE, GetLeaves_I007)
{
	FUZZ::reset_logger();
	size_t nnodes = FUZZ::getInt(1, "nnodes", nnodes_range)[0];

	std::unordered_set<variable<double>*> leaves;
	std::unordered_set<immutable<double>*> collector;

	inode<double>* root = FUZZ::buildNTree<inode<double> >(2, nnodes,
		[&leaves]() -> inode<double>*
		{
			std::string llabel = FUZZ::getString(FUZZ::getInt(1, "llabel.size", {14, 29})[0], "llabel");
			double leafvalue = FUZZ::getDouble(1, "leafvalue")[0];
			variable<double>* im = new variable<double>(leafvalue, llabel);
			leaves.emplace(im);
			return im;
		},
		[&collector](std::vector<inode<double>*> args) -> inode<double>*
		{
			std::string nlabel = FUZZ::getString(FUZZ::getInt(1, "nlabel.size", {14, 29})[0], "nlabel");
			mock_immutable* im = new mock_immutable(
				std::vector<inode<double>*>(args.begin(), args.end()), nlabel);
			im->triggerOnDeath =
				[&collector](mock_immutable* ded) {
					collector.erase(ded);
				};
			collector.insert(im);
			return im;
		});

	// the root has all leaves
	std::unordered_set<ileaf<double>*> lcache = root->get_leaves();
	for (variable<double>* l : leaves)
	{
		EXPECT_TRUE(lcache.end() != lcache.find(l));
	}
	// any collector's leaf is found in leaves (ensures lcache doesn't collect trash nodes)
	for (immutable<double>* coll : collector)
	{
		lcache.clear();
		lcache = coll->get_leaves();
		for (ileaf<double>* useful : lcache)
		{
			if (variable<double>* uvar = dynamic_cast<variable<double>*>(useful))
			{
				EXPECT_TRUE(leaves.end() != leaves.find(uvar));
			}
		}
	}

	for (variable<double>* l : leaves)
	{
		delete l;
	}
	for (immutable<double>* im : collector)
	{
		delete im;
	}
}


TEST(IMMUTABLE, GetLeaf_I008)
{
	FUZZ::reset_logger();
	std::vector<iconnector<double>*> ordering;
	mock_node exposer;

	BACK_MAP<double> backer =
	[&ordering](std::vector<std::pair<inode<double>*,inode<double>*> > args) -> inode<double>*
	{
		inode<double>* leef = args[0].second;
		double lvalue = expose<double>(leef)[0];
		if (lvalue == 0.0 && args.size() > 1)
		{
			leef = args[1].second;
			if (iconnector<double>* conn = dynamic_cast<iconnector<double>*>(args[1].first))
			{
				ordering.push_back(conn);
			}
		}
		else if (iconnector<double>* conn = dynamic_cast<iconnector<double>*>(args[0].first))
		{
			ordering.push_back(conn);
		}
		return leef;
	};

	size_t nnodes = FUZZ::getInt(1, "nnodes", nnodes_range)[0];

	std::unordered_set<variable<double>*> leaves;
	std::unordered_set<immutable<double>*> collector;

	inode<double>* root = FUZZ::buildNTree<inode<double> >(2, nnodes,
		[&leaves]() -> inode<double>*
		{
			std::string llabel = FUZZ::getString(FUZZ::getInt(1, "llabel.size", {14, 29})[0], "llabel");
			double leafvalue = FUZZ::getDouble(1, "leafvalue")[0];
			variable<double>* im = new variable<double>(leafvalue, llabel);
			leaves.emplace(im);
			return im;
		},
		[&collector, &backer](std::vector<inode<double>*> args) -> inode<double>*
		{
			std::string nlabel = FUZZ::getString(FUZZ::getInt(1, "nlabel.size", {14, 29})[0], "nlabel");
			mock_immutable* im = new mock_immutable(args, nlabel,
				get_testshaper(), testtrans, backer);
			im->triggerOnDeath =
				[&collector](mock_immutable* ded) {
					collector.erase(ded);
				};
			collector.insert(im);
			return im;
		});

	variable<double>* notleaf = new variable<double>(0);
	for (size_t i = 0; i < nnodes/3; i++)
	{
		ordering.clear();
		variable<double>* l = *(FUZZ::rand_select<std::unordered_set<variable<double>*> >(leaves));
		varptr<double> wun = exposer.expose_leaf(root, l);
		EXPECT_TRUE(bottom_up(ordering));
		ordering.clear();
		varptr<double> zaro = exposer.expose_leaf(root, notleaf);
		EXPECT_TRUE(bottom_up(ordering));

		double value1 = expose<double>(wun)[0];
		double value0 = expose<double>(zaro)[0];
		EXPECT_TRUE(value1 == 1.0);
		EXPECT_TRUE(value0 == 0.0);
	}

	delete notleaf;
	for (variable<double>* l : leaves)
	{
		delete l;
	}
	for (immutable<double>* im : collector)
	{
		delete im;
	}
}


TEST(IMMUTABLE, GetGradient_I009)
{
	FUZZ::reset_logger();
	tensorshape shape = random_def_shape();
	double single_rando = FUZZ::getDouble(1, "single_rando", {1.1, 2.2})[0];

	auto unifiedshaper =
	[&shape](std::vector<tensorshape>)
	{
		return shape;
	};

	const_init<double> cinit(single_rando);

	std::vector<iconnector<double>*> ordering;
	BACK_MAP<double> backer =
	[&ordering](std::vector<std::pair<inode<double>*,inode<double>*> > args) -> inode<double>*
	{
		varptr<double> leef = args[0].second;
		double d = expose<double>(leef.get())[0];
		if (d == 0.0 && args.size() > 1)
		{
			leef = args[1].second;
			if (iconnector<double>* conn = dynamic_cast<iconnector<double>*>(args[1].first))
			{
				ordering.push_back(conn);
			}
		}
		else if (iconnector<double>* conn = dynamic_cast<iconnector<double>*>(args[0].first))
		{
			ordering.push_back(conn);
		}
		return leef;
	};

	size_t nnodes = FUZZ::getInt(1, "nnodes", nnodes_range)[0];

	std::unordered_set<variable<double>*> leaves;
	std::unordered_set<immutable<double>*> collector;

	inode<double>* root = FUZZ::buildNTree<inode<double> >(2, nnodes,
	[&leaves, &shape, &cinit]() -> inode<double>*
	{
		std::string llabel = FUZZ::getString(FUZZ::getInt(1, "llabel.size", {14, 29})[0], "llabel");
		variable<double>* im = new variable<double>(shape, cinit, llabel);
		im->initialize();
		leaves.emplace(im);
		return im;
	},
	[&collector, &unifiedshaper, &backer](std::vector<inode<double>*> args) -> inode<double>*
	{
		std::string nlabel = FUZZ::getString(FUZZ::getInt(1, "nlabel.size", {14, 29})[0], "nlabel");
		mock_immutable* im = new mock_immutable(args, nlabel,
			unifiedshaper, adder, backer);
		im->triggerOnDeath =
			[&collector](mock_immutable* ded) {
				collector.erase(ded);
			};
		collector.insert(im);
		return im;
	});

	variable<double>* notleaf = new variable<double>(0);
	std::unordered_set<ileaf<double>*> lcache;
	for (size_t i = 0; i < nnodes/3; i++)
	{
		ordering.clear();
		variable<double>* rselected = *(FUZZ::rand_select<std::unordered_set<variable<double>*> >(leaves));
		const tensor<double>* wun = root->derive(rselected)->eval();
		EXPECT_TRUE(bottom_up(ordering));
		ordering.clear();
		const tensor<double>* zaro = root->derive(notleaf)->eval();
		EXPECT_TRUE(bottom_up(ordering));
		ordering.clear();

		ASSERT_NE(nullptr, wun);
		ASSERT_NE(nullptr, zaro);
		EXPECT_EQ(1, wun->expose()[0]);
		EXPECT_EQ(0, zaro->expose()[0]);

		// SAME AS TEMPORARY EVAL
		immutable<double>* coll = *(FUZZ::rand_select<std::unordered_set<immutable<double>*> >(collector));
		if (coll == root) continue;
		const tensor<double>* grad_too = root->derive(coll)->eval();
		EXPECT_TRUE(bottom_up(ordering));
		ASSERT_NE(nullptr, grad_too);
		ASSERT_TRUE(tensorshape_equal(shape, grad_too->get_shape()));
		// out data should be 1 + M * single_rando where M is the
		// number of root's leaves that are not in coll's leaves
		lcache.clear();
		lcache = coll->get_leaves();
		size_t M = leaves.size() - lcache.size();
		double datum = M * single_rando + 1;
		std::vector<double> odata = grad_too->expose();
		double diff = std::abs(datum - odata[0]);
		EXPECT_GT(0.000001 * single_rando, diff); // allow error of a tiny fraction of the random leaf value
	}

	delete notleaf;
	for (variable<double>* l : leaves)
	{
		delete l;
	}
	for (immutable<double>* im : collector)
	{
		delete im;
	}
}


TEST(IMMUTABLE, Update_I010)
{
	FUZZ::reset_logger();
	std::string conname = FUZZ::getString(FUZZ::getInt(1, "conname.size", {14, 29})[0], "conname");
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");

	mock_node* n1 = new mock_node(label1);
	tensorshape n1s = random_def_shape();
	n1->data_ = new mock_tensor(n1s);

	// for this test, we care about data, grab the largest shape, and sum all data that fit in said array
	auto grabs = [](std::vector<tensorshape> ts)
	{
		return ts[0];
	};

	bool mutate = false;
	TRANSFER_FUNC<double> asis = [&mutate](double* dest, std::vector<const double*> src, nnet::shape_io shape)
	{
		adder(dest, src, shape);
		if (mutate)
		{
			for (size_t i = 0; i < shape.outs_.n_elems(); i++)
			{
				dest[i] += src.size();
			}
		}
	};

	immutable<double>* conn = new mock_immutable({n1}, conname, grabs, asis);
	std::vector<double> init = expose(conn);
	mutate = true;
	conn->update(std::unordered_set<size_t>{});
	std::vector<double> next = expose(conn);
	ASSERT_EQ(init.size(), next.size());
	for (size_t i = 0, n = init.size(); i < n; i++)
	{
		EXPECT_EQ(init[i]+1, next[i]);
	}

	delete conn;
	delete n1;
}


TEST(IMMUTABLE, ShapeIncompatible_I011)
{
	FUZZ::reset_logger();
	std::string conname = FUZZ::getString(FUZZ::getInt(1, "conname.size", {14, 29})[0], "conname");
	std::string conname2 = FUZZ::getString(FUZZ::getInt(1, "conname2.size", {14, 29})[0], "conname2");
	std::string label1 = FUZZ::getString(FUZZ::getInt(1, "label1.size", {14, 29})[0], "label1");

	mock_node* n1 = new mock_node(label1);
	tensorshape n1s = random_def_shape();
	std::vector<size_t> temp = n1s.as_list();
	temp.push_back(3);
	tensorshape n2s = temp;
	n1->data_ = new mock_tensor(n1s);

	bool change = false;
	auto shiftyshaper =
	[&change, n2s](std::vector<tensorshape> ts)
	{
		if (change) return n2s;
		return ts[0];
	};

	mock_immutable* initialgood = new mock_immutable({n1}, conname2, shiftyshaper);
	change = true;
	EXPECT_THROW(n1->notify(nnet::notification::UPDATE), std::exception);

	delete initialgood;
	delete n1;
}


#endif /* DISABLE_IMMUTABLE_TEST */


#endif /* DISABLE_CONNECTOR_MODULE_TESTS */
