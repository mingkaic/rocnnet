//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "graph/connector/immutable/matmul.hpp"

#include "gtest/gtest.h"
#include "util_test.h"
#include "fuzz.h"


#ifndef DISABLE_MATMUL_TEST


using namespace nnet;


TEST(MATMUL, Copy_L000)
{
	FUZZ::reset_logger();
	tensorshape shapeA = random_def_shape(2, 2);
	std::vector<size_t> alist = shapeA.as_list();
	tensorshape shapeB(std::vector<size_t>{alist[1], alist[0]});
	rand_uniform<double> rinit(2, 12);
	constant<double>* zero = constant<double>::get(0);

	matmul<double>* olassign = matmul<double>::get(zero, zero);
	matmul<double>* bothassign = matmul<double>::get(zero, zero);

	variable<double> A(shapeA, rinit, "A"); // shape <m, k>
	variable<double> B(shapeB, rinit, "B"); // shape <k, m>

	matmul<double>* olnatural = matmul<double>::get(&A, &B); // shape <m, m>
	matmul<double>* bothtrans = matmul<double>::get(&A, &B, true, true); // shape <k, k>

	matmul<double>* olcpy = olnatural->clone();
	matmul<double>* bothcpy = bothtrans->clone();

	*olassign = *olnatural;
	*bothassign = *bothtrans;

	A.initialize();
	B.initialize();

	EXPECT_TRUE(tensorshape_equal(olnatural->get_shape(), olcpy->get_shape()));
	EXPECT_TRUE(tensorshape_equal(bothtrans->get_shape(), bothcpy->get_shape()));
	EXPECT_TRUE(tensorshape_equal(olnatural->get_shape(), olassign->get_shape()));
	EXPECT_TRUE(tensorshape_equal(bothtrans->get_shape(), bothassign->get_shape()));

	std::vector<double> naturaldata = expose(olnatural);
	std::vector<double> bothdata = expose(bothtrans);
	std::vector<double> ocpydata = expose(olcpy);
	std::vector<double> bcpyata = expose(bothcpy);
	std::vector<double> oassigndata = expose(olassign);
	std::vector<double> bassigndata = expose(bothassign);

	EXPECT_TRUE(std::equal(naturaldata.begin(), naturaldata.end(), ocpydata.begin()));
	EXPECT_TRUE(std::equal(bothdata.begin(), bothdata.end(), bcpyata.begin()));
	EXPECT_TRUE(std::equal(naturaldata.begin(), naturaldata.end(), oassigndata.begin()));
	EXPECT_TRUE(std::equal(bothdata.begin(), bothdata.end(), bassigndata.begin()));

	delete olnatural;
	delete bothtrans;
	delete olcpy;
	delete bothcpy;
	delete olassign;
	delete bothassign;
}


TEST(MATMUL, Move_L000)
{
	FUZZ::reset_logger();
	tensorshape shapeA = random_def_shape(2, 2);
	std::vector<size_t> alist = shapeA.as_list();
	tensorshape shapeB(std::vector<size_t>{alist[1], alist[0]});
	rand_uniform<double> rinit(2, 12);
	constant<double>* zero = constant<double>::get(0);

	matmul<double>* olassign = matmul<double>::get(zero, zero);
	matmul<double>* bothassign = matmul<double>::get(zero, zero);

	variable<double> A(shapeA, rinit, "A"); // shape <m, k>
	variable<double> B(shapeB, rinit, "B"); // shape <k, m>

	matmul<double>* olnatural = matmul<double>::get(&A, &B); // shape <m, m>
	matmul<double>* bothtrans = matmul<double>::get(&A, &B, true, true); // shape <k, k>

	tensorshape oshape = olnatural->get_shape();
	tensorshape bshape = bothtrans->get_shape();
	std::vector<double> naturaldata = expose(olnatural);
	std::vector<double> bothdata = expose(bothtrans);

	matmul<double>* olmv = olnatural->move();
	matmul<double>* bothmv = bothtrans->move();

	EXPECT_TRUE(tensorshape_equal(oshape, olmv->get_shape()));
	EXPECT_TRUE(tensorshape_equal(bshape, bothmv->get_shape()));
	EXPECT_FALSE(olnatural->good_status());
	EXPECT_FALSE(bothtrans->good_status());
	std::vector<double> omvdata = expose(olmv);
	std::vector<double> bmvata = expose(bothmv);
	EXPECT_TRUE(std::equal(naturaldata.begin(), naturaldata.end(), omvdata.begin()));
	EXPECT_TRUE(std::equal(bothdata.begin(), bothdata.end(), bmvata.begin()));

	*olassign = std::move(*olmv);
	*bothassign = std::move(*bothmv);

	EXPECT_TRUE(tensorshape_equal(oshape, olassign->get_shape()));
	EXPECT_TRUE(tensorshape_equal(bshape, bothassign->get_shape()));
	EXPECT_FALSE(olmv->good_status());
	EXPECT_FALSE(bothmv->good_status());
	std::vector<double> oassigndata = expose(olassign);
	std::vector<double> bassigndata = expose(bothassign);
	EXPECT_TRUE(std::equal(naturaldata.begin(), naturaldata.end(), oassigndata.begin()));
	EXPECT_TRUE(std::equal(bothdata.begin(), bothdata.end(), bassigndata.begin()));

	delete olnatural;
	delete bothtrans;
	delete olmv;
	delete bothmv;
	delete olassign;
	delete bothassign;
}


TEST(MATMUL, NullptrRet_L001)
{
	FUZZ::reset_logger();
	constant<double>* zero = constant<double>::get(0);
	EXPECT_EQ(nullptr, matmul<double>::get(nullptr, nullptr));
	EXPECT_EQ(nullptr, matmul<double>::get(zero, nullptr));
	EXPECT_EQ(nullptr, matmul<double>::get(nullptr, zero));
	delete zero;
}


TEST(MATMUL, DISABLED_Matmul_L002)
{
	FUZZ::reset_logger();
}


TEST(MATMUL, DISABLED_Incompatible_L003)
{
	FUZZ::reset_logger();
}


TEST(MATMUL, DISABLED_Jacobian_L004)
{
	FUZZ::reset_logger();
}


#endif /* DISABLE_MATMUL_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
