//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATION_MODULE_TESTS

#include <algorithm>

#include "graph/operations/operations.hpp"
#include "utils/futils.hpp"

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


TEST(MATMUL, NullptrRet_C000)
{
	FUZZ::reset_logger();
	variable<double>* zero = new variable<double>(0);
	EXPECT_EQ(nullptr, matmul<double>(nullptr, nullptr));
	EXPECT_EQ(nullptr, matmul<double>(zero, nullptr));
	EXPECT_EQ(nullptr, matmul<double>(nullptr, zero));
	delete zero;
}


TEST(MATMUL, Matmul_C001)
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
	varptr<signed> res = matmul<signed>(varptr<signed>(&A), varptr<signed>(&B));
	varptr<signed> restA = matmul<signed>(varptr<signed>(&tA), varptr<signed>(&B), true);
	varptr<signed> restB = matmul<signed>(varptr<signed>(&A), varptr<signed>(&tB), false, true);
	varptr<signed> resT = matmul<signed>(varptr<signed>(&tA), varptr<signed>(&tB), true, true);

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

	TWODV matA = create2D(expose<signed>(&A), A.get_shape());
	TWODV matB = create2D(expose<signed>(&B), B.get_shape());
	TWODV mattA = create2D(expose<signed>(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose<signed>(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose<signed>(res), resshape);
	TWODV matrestA = create2D(expose<signed>(restA), restAshape);
	TWODV matrestB = create2D(expose<signed>(restB), restBshape);
	TWODV matresT = create2D(expose<signed>(resT), resTshape);

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
// todo: test
TEST(MATMUL, DISABLED_NDim_Matmul_C001)
{
	FUZZ::reset_logger();
}


TEST(MATMUL, Incompatible_C002)
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

	varptr<signed> bad = matmul<signed>(varptr<signed>(&A), varptr<signed>(&B));
	EXPECT_THROW(bad->eval(), std::logic_error);
}


TEST(MATMUL, Jacobian_C003)
{
	FUZZ::reset_logger();
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = FUZZ::getInt(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform<double> rinit(0, 1);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable<double> A(shapeA, rinit, "A"); // shape <m, n>
	variable<double> B(shapeB, rinit, "B"); // shape <k, m>
	variable<double> tA(shapetA, rinit, "tA");
	variable<double> tB(shapetB, rinit, "tB");

	// shapes of <k, n>
	varptr<double> res = sigmoid(varptr<double>(matmul<double>(varptr<double>(&A), varptr<double>(&B))));
	varptr<double> restA = sigmoid(varptr<double>(matmul<double>(varptr<double>(&tA), varptr<double>(&B), true)));
	varptr<double> restB = sigmoid(varptr<double>(matmul<double>(varptr<double>(&A), varptr<double>(&tB), false, true)));
	varptr<double> resT = sigmoid(varptr<double>(matmul<double>(varptr<double>(&tA), varptr<double>(&tB), true, true)));

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	inode<double>* dresA = res->derive(&A);
	inode<double>* dresB = res->derive(&B);

	inode<double>* drestAA = restA->derive(&tA);
	inode<double>* drestAB = restA->derive(&B);

	inode<double>* drestBA = restB->derive(&A);
	inode<double>* drestBB = restB->derive(&tB);

	inode<double>* dresTA = resT->derive(&tA);
	inode<double>* dresTB = resT->derive(&tB);

	// requires on all elementary operations to be valid (not a great validation method...)
	// res = 1/(1+e^-(A@B))
	// dres = jacobian(sigmoid'(1))
	// where jacobian = {
	// 		sigmoid'(1) @ B^T for dA
	//		A^T @ sigmoid'(1) for dB
	// }
	// sigmoid' = sigmoid * (1 - sigmoid)
	varptr<double> dsig_res = res * (1.0 - res);
	inode<double>* fake_dresA = matmul<double>(dsig_res, &B, false, true);
	inode<double>* fake_dresB = matmul<double>(&A, dsig_res, true);

	varptr<double> dsig_restA = restA * (1.0 - restA);
	inode<double>* fake_drestAA = transpose<double>(matmul<double>(dsig_restA, &B, false, true));
	inode<double>* fake_drestAB = matmul<double>(&tA, dsig_restA);

	varptr<double> dsig_restB = restB * (1.0 - restB);
	inode<double>* fake_drestBA = matmul<double>(dsig_restB, &tB);
	inode<double>* fake_drestBB = transpose<double>(matmul<double>(&A, dsig_restB, true, false));

	varptr<double> dsig_resT = resT * (1.0 - resT);
	inode<double>* fake_dresTA = transpose<double>(matmul<double>(dsig_resT, &tB));
	inode<double>* fake_dresTB = transpose<double>(matmul<double>(&tA, dsig_resT));

	EXPECT_TRUE(tensorshape_equal(dresA->get_shape(), A.get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresB->get_shape(), B.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAA->get_shape(), tA.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAB->get_shape(), B.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBA->get_shape(), A.get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBB->get_shape(), tB.get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTA->get_shape(), tA.get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTB->get_shape(), tB.get_shape()));

	EXPECT_TRUE(tensorshape_equal(dresA->get_shape(), fake_dresA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresB->get_shape(), fake_dresB->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAA->get_shape(), fake_drestAA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestAB->get_shape(), fake_drestAB->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBA->get_shape(), fake_drestBA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(drestBB->get_shape(), fake_drestBB->get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTA->get_shape(), fake_dresTA->get_shape()));
	EXPECT_TRUE(tensorshape_equal(dresTB->get_shape(), fake_dresTB->get_shape()));

	std::vector<double> dresA_data = expose<double>(dresA);
	std::vector<double> dresB_data = expose<double>(dresB);
	std::vector<double> drestAA_data = expose<double>(drestAA);
	std::vector<double> drestAB_data = expose<double>(drestAB);
	std::vector<double> drestBA_data = expose<double>(drestBA);
	std::vector<double> drestBB_data = expose<double>(drestBB);
	std::vector<double> dresTA_data = expose<double>(dresTA);
	std::vector<double> dresTB_data = expose<double>(dresTB);

	std::vector<double> fake_dresA_data = expose<double>(fake_dresA);
	std::vector<double> fake_dresB_data = expose<double>(fake_dresB);
	std::vector<double> fake_drestAA_data = expose<double>(fake_drestAA);
	std::vector<double> fake_drestAB_data = expose<double>(fake_drestAB);
	std::vector<double> fake_drestBA_data = expose<double>(fake_drestBA);
	std::vector<double> fake_drestBB_data = expose<double>(fake_drestBB);
	std::vector<double> fake_dresTA_data = expose<double>(fake_dresTA);
	std::vector<double> fake_dresTB_data = expose<double>(fake_dresTB);

	// all a shapes should have the same number of elements
	double err_thresh = 0.0000001;
	for (size_t i = 0, n = dresA_data.size(); i < n; i++)
	{
		double dresAerr = std::abs(dresA_data[i] - fake_dresA_data[i]);
		double drestAAerr = std::abs(drestAA_data[i] - fake_drestAA_data[i]);
		double drestBAerr = std::abs(drestBA_data[i] - fake_drestBA_data[i]);
		double dresTAerr = std::abs(dresTA_data[i] - fake_dresTA_data[i]);
		EXPECT_GT(err_thresh, dresAerr);
		EXPECT_GT(err_thresh, drestAAerr);
		EXPECT_GT(err_thresh, drestBAerr);
		EXPECT_GT(err_thresh, dresTAerr);
	}
	for (size_t i = 0, n = dresB_data.size(); i < n; i++)
	{
		double dresBerr = std::abs(dresB_data[i] - fake_dresB_data[i]);
		double drestABerr = std::abs(drestAB_data[i] - fake_drestAB_data[i]);
		double drestBBerr = std::abs(drestBB_data[i] - fake_drestBB_data[i]);
		double dresTBerr = std::abs(dresTB_data[i] - fake_dresTB_data[i]);
		EXPECT_GT(err_thresh, dresBerr);
		EXPECT_GT(err_thresh, drestABerr);
		EXPECT_GT(err_thresh, drestBBerr);
		EXPECT_GT(err_thresh, dresTBerr);
	}
}


// tests large matrices sizes (100-112), 2D only
TEST(MATMUL, Strassen_C004)
{
	FUZZ::reset_logger();
	// we get at most 12996 elements per matrix
	std::vector<size_t> dims = FUZZ::getInt(3, "dimensions<m,n,k>", {STRASSEN_THRESHOLD, STRASSEN_THRESHOLD+12});
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
//	clock_t t = clock();
	varptr<signed> res = matmul<signed>(varptr<signed>(&A), varptr<signed>(&B));
//	const double work_time1 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr<signed> restA = matmul<signed>(varptr<signed>(&tA), varptr<signed>(&B), true);
//	const double work_time2 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr<signed> restB = matmul<signed>(varptr<signed>(&A), varptr<signed>(&tB), false, true);
//	const double work_time3 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr<signed> resT = matmul<signed>(varptr<signed>(&tA), varptr<signed>(&tB), true, true);
//	const double work_time4 = (clock() - t) / double(CLOCKS_PER_SEC);
//	ASSERT_GT(0.3, work_time1);
//	ASSERT_GT(0.3, work_time2);
//	ASSERT_GT(0.3, work_time3);
//	ASSERT_GT(0.3, work_time4);

	tensorshape expectshape = std::vector<size_t>{dims[2], dims[1]};
	tensorshape resshape = res->get_shape();
	tensorshape restAshape = restA->get_shape();
	tensorshape restBshape = restB->get_shape();
	tensorshape resTshape = resT->get_shape();

	ASSERT_TRUE(tensorshape_equal(expectshape, resshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restAshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, restBshape));
	ASSERT_TRUE(tensorshape_equal(expectshape, resTshape));

	TWODV matA = create2D(expose<signed>(&A), A.get_shape());
	TWODV matB = create2D(expose<signed>(&B), B.get_shape());
	TWODV mattA = create2D(expose<signed>(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose<signed>(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose<signed>(res), resshape);
	TWODV matrestA = create2D(expose<signed>(restA), restAshape);
	TWODV matrestB = create2D(expose<signed>(restB), restBshape);
	TWODV matresT = create2D(expose<signed>(resT), resTshape);
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


#endif /* DISABLE_OPERATION_MODULE_TESTS */
