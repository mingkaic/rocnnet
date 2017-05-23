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


using TWODV = std::vector<std::vector<signed> >;


TWODV create2D (std::vector<signed> juanD, tensorshape mats, bool transpose = false)
{
	std::vector<size_t> dims = mats.as_list();
	size_t C = dims[0];
	size_t R = dims[1];
	TWODV res;

	size_t resC = transpose ? R : C;
	size_t resR = transpose ? C : R;
 	for (size_t y = 0; y < resR; y++)
	{
		res.push_back(std::vector<signed>(resC, 0));
	}

	for (size_t y = 0; y < R; y++)
	{
		for (size_t x = 0; x < C; x++)
		{
			size_t juan_coord = x + y * C;
			if (transpose)
			{
				res[x][y] = juanD[juan_coord];
			}
			else
			{
				res[y][x] = juanD[juan_coord];
			}
		}
	}
	return res;
}


bool freivald (TWODV a, TWODV b, TWODV c)
{
	assert(!b.empty());
	size_t rlen = b[0].size();
	// probability of false positive = 1/2^n
	// Pr(fp) = 0.1% ~~> n = 10
	size_t m = 10;
	for (size_t i = 0; i < m; i++)
	{
		// generate r of len b[0].size() or c[0].size()
		std::vector<size_t> r = FUZZ::getInt(rlen, nnutils::formatter() << "freivald_vec" << i, {0, 1});

		// p = a @ (b @ r) - c @ r
		std::vector<signed> br;
		for (size_t y = 0, n = b.size(); y < n; y++)
		{
			signed bri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				bri += b[y][x] * r[x];
			}
			br.push_back(bri);
		}

		std::vector<signed> cr;
		for (size_t y = 0, n = c.size(); y < n; y++)
		{
			signed cri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				cri += c[y][x] * r[x];
			}
			cr.push_back(cri);
		}

		std::vector<signed> p;
		size_t n = a.size();
		for (size_t y = 0; y < n; y++)
		{
			signed ari = 0;
			for (size_t x = 0, m = a[y].size(); x < m; x++)
			{
				ari += a[y][x] * br[x];
			}
			p.push_back(ari);
		}
		for (size_t j = 0; j < n; j++)
		{
			p[j] -= cr[j];
		}

		// if p != 0 -> return false
		if (!std::all_of(p.begin(), p.end(), [](signed d) { return d == 0; }))
			return false;
	}
	return true;
}


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


TEST(MATMUL, Matmul_L002)
{
	FUZZ::reset_logger();
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = FUZZ::getInt(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform<signed> rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable<signed> A(shapeA, rinit, "A"); // shape <m, n>
	variable<signed> B(shapeB, rinit, "B"); // shape <k, m>
	variable<signed> tA(shapetA, rinit, "tA");
	variable<signed> tB(shapetB, rinit, "tB");

	// shapes of <k, n>
	matmul<signed>* res = matmul<signed>::get(&A, &B);
	matmul<signed>* restA = matmul<signed>::get(&tA, &B, true);
	matmul<signed>* restB = matmul<signed>::get(&A, &tB, false, true);
	matmul<signed>* resT = matmul<signed>::get(&tA, &tB, true, true);

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	tensorshape expectshape = std::vector<size_t>{dims[2], dims[1]};
	tensorshape resshape = res->get_shape();
	tensorshape restAshape = restA->get_shape();
	tensorshape restBshape = restB->get_shape();
	tensorshape resTshape = resT->get_shape();

	ASSERT_TRUE(tensorshape_equal(expectshape, resshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restAshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restBshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, resTshape));

	TWODV matA = create2D(expose(&A), A.get_shape());
	TWODV matB = create2D(expose(&B), B.get_shape());
	TWODV mattA = create2D(expose(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose(res), resshape);
	TWODV matrestA = create2D(expose(restA), restAshape);
	TWODV matrestB = create2D(expose(restB), restBshape);
	TWODV matresT = create2D(expose(resT), resTshape);
	// Freivald's algorithm

	EXPECT_TRUE(freivald(matA, matB, matres));
	EXPECT_TRUE(freivald(mattA, matB, matrestA));
	EXPECT_TRUE(freivald(matA, mattB, matrestB));
	EXPECT_TRUE(freivald(mattA, mattB, matresT));

	// we delete top nodes, because this case is not testing for observer self-destruction
	delete res;
	delete restA;
	delete restB;
	delete resT;
}


// tests matrix multiplication but for n dimensions, matrix sizes reduced to 2-5, (we get at most 5x25 matmuls)
TEST(MATMUL, DISABLED_NDim_Matmul_L002)
{
	FUZZ::reset_logger();
}


TEST(MATMUL, Incompatible_L003)
{
	FUZZ::reset_logger();
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = FUZZ::getInt(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform<signed> rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]+1};

	variable<signed> A(shapeA, rinit, "A"); // shape <m, n>
	variable<signed> B(shapeB, rinit, "B"); // shape <k, m+1>

	A.initialize();
	B.initialize();

	EXPECT_THROW(matmul<signed>::get(&A, &B), std::logic_error);
}


TEST(MATMUL, DISABLED_Jacobian_L004)
{
	FUZZ::reset_logger();
}


// tests large matrices sizes (100-112), 2D only
TEST(MATMUL, Strassen_L005)
{
	FUZZ::reset_logger();
	// we get at most 12996 elements per matrix
	std::vector<size_t> dims = FUZZ::getInt(3, "dimensions<m,n,k>", {100, 112});
	rand_uniform<signed> rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable<signed> A(shapeA, rinit, "A"); // shape <m, n>
	variable<signed> B(shapeB, rinit, "B"); // shape <k, m>
	variable<signed> tA(shapetA, rinit, "tA");
	variable<signed> tB(shapetB, rinit, "tB");

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	// shapes of <k, n>
	clock_t t = clock();
	matmul<signed>* res = matmul<signed>::get(&A, &B);
	const double work_time1 = (clock() - t) / double(CLOCKS_PER_SEC);

	t = clock();
	matmul<signed>* restA = matmul<signed>::get(&tA, &B, true);
	const double work_time2 = (clock() - t) / double(CLOCKS_PER_SEC);

	t = clock();
	matmul<signed>* restB = matmul<signed>::get(&A, &tB, false, true);
	const double work_time3 = (clock() - t) / double(CLOCKS_PER_SEC);

	t = clock();
	matmul<signed>* resT = matmul<signed>::get(&tA, &tB, true, true);
	const double work_time4 = (clock() - t) / double(CLOCKS_PER_SEC);
	ASSERT_LT(0.003, work_time1);
	ASSERT_LT(0.003, work_time2);
	ASSERT_LT(0.003, work_time3);
	ASSERT_LT(0.003, work_time4);

	tensorshape expectshape = std::vector<size_t>{dims[2], dims[1]};
	tensorshape resshape = res->get_shape();
	tensorshape restAshape = restA->get_shape();
	tensorshape restBshape = restB->get_shape();
	tensorshape resTshape = resT->get_shape();

	ASSERT_TRUE(tensorshape_equal(expectshape, resshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restAshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restBshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, resTshape));

	TWODV matA = create2D(expose(&A), A.get_shape());
	TWODV matB = create2D(expose(&B), B.get_shape());
	TWODV mattA = create2D(expose(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose(res), resshape);
	TWODV matrestA = create2D(expose(restA), restAshape);
	TWODV matrestB = create2D(expose(restB), restBshape);
	TWODV matresT = create2D(expose(resT), resTshape);
	// Freivald's algorithm

	EXPECT_TRUE(freivald(matA, matB, matres));
	EXPECT_TRUE(freivald(mattA, matB, matrestA));
	EXPECT_TRUE(freivald(matA, mattB, matrestB));
	EXPECT_TRUE(freivald(mattA, mattB, matresT));

	// we delete top nodes, because this case is not testing for observer self-destruction
	delete res;
	delete restA;
	delete restB;
	delete resT;
}


#endif /* DISABLE_MATMUL_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
