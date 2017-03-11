//
// Created by Mingkai Chen on 2016-08-29.
//

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"

#include "mocks/mock_tensor.h"


//#define DISABLE_TENSOR_TEST
#ifndef DISABLE_TENSOR_TEST


static tensorshape random_partialshape ()
{
	size_t rank = FUZZ<size_t>::get(1, {1, 5})[0];
	size_t nzeros = FUZZ<size_t>::get(1, {1, 5})[0];
	std::vector<size_t> shape = FUZZ<size_t>::get(rank, {2, 21});
	for (size_t i = 0; i < nzeros; i++)
	{
		size_t zidx = FUZZ<size_t>::get(1, {0, shape.size()})[0];
		shape.insert(shape.begin()+zidx, 0);
	}
	return tensorshape(shape);
}


static tensorshape random_shape ()
{
	size_t rank = FUZZ<size_t>::get(1, {2, 10})[0];
	std::vector<size_t> shape = FUZZ<size_t>::get(rank, {2, 21});
	return tensorshape(shape);
}


// cover tensor
// default, scalar, shape constructors,
// is_alloc, total_bytes
TEST(TENSOR, Construct_B000)
{
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_shape();

	mock_tensor undef;
	mock_tensor scalar(FUZZ<double>::get(1)[0]);
	mock_tensor incom(pshape);
	mock_tensor comp(cshape);

	EXPECT_TRUE(undef.clean());
	EXPECT_TRUE(scalar.clean());
	EXPECT_TRUE(incom.clean());
	EXPECT_TRUE(comp.clean());

	EXPECT_FALSE(undef.is_alloc());
	EXPECT_TRUE(scalar.is_alloc());
	EXPECT_FALSE(incom.is_alloc());
	EXPECT_TRUE(comp.is_alloc());

	EXPECT_EQ(0, undef.total_bytes());
	EXPECT_EQ(sizeof(double), scalar.total_bytes());
	EXPECT_EQ(0, incom.total_bytes());
	EXPECT_EQ(sizeof(double) * cshape.n_elems(),
		comp.total_bytes());
}


// cover tensor
// clone and assignment
TEST(TENSOR, Copy_B001)
{
	mock_tensor undefassign;
	mock_tensor scalarassign;
	mock_tensor incomassign;
	mock_tensor compassign;

	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_shape();

	mock_tensor undef;
	mock_tensor scalar(FUZZ<double>::get(1)[0]);
	mock_tensor incom(pshape);
	mock_tensor comp(cshape);

	mock_tensor* undefcpy = undef.clone();
	mock_tensor* scalarcpy = scalar.clone();
	mock_tensor* incomcpy = incom.clone();
	mock_tensor* compcpy = comp.clone();
	undefassign = undef;
	scalarassign = scalar;
	incomassign = incom;
	compassign = comp;

	EXPECT_FALSE(undefcpy->is_alloc());
	EXPECT_TRUE(scalarcpy->is_alloc());
	EXPECT_FALSE(incomcpy->is_alloc());
	EXPECT_TRUE(compcpy->is_alloc());
	EXPECT_FALSE(undefassign.is_alloc());
	EXPECT_TRUE(scalarassign.is_alloc());
	EXPECT_FALSE(incomassign.is_alloc());
	EXPECT_TRUE(compassign.is_alloc());

	EXPECT_TRUE(undefcpy->equal(undef));
	EXPECT_TRUE(scalarcpy->equal(scalar));
	EXPECT_TRUE(incomcpy->equal(incom));
	EXPECT_TRUE(compcpy->equal(comp));
	EXPECT_TRUE(undefassign.equal(undef));
	EXPECT_TRUE(scalarassign.equal(scalar));
	EXPECT_TRUE(incomassign.equal(incom));
	EXPECT_TRUE(compassign.equal(comp));

	delete undefcpy;
	delete scalarcpy;
	delete incomcpy;
	delete compcpy;
}


// cover tensor
// move constructor and assignment
TEST(TENSOR, Move_B001)
{
	mock_tensor scalarassign;
	mock_tensor compassign;

	tensorshape sshape(std::vector<size_t>{1});
	tensorshape cshape = random_shape();
	mock_tensor scalar(FUZZ<double>::get(1)[0]);
	mock_tensor comp(cshape);

	double* scalarptr = scalar.rawptr();
	double* compptr = comp.rawptr();

	mock_tensor scalarmv(std::move(scalar));
	mock_tensor compmv(std::move(comp));

	EXPECT_FALSE(scalar.is_alloc());
	EXPECT_FALSE(comp.is_alloc());
	EXPECT_EQ(scalarptr, scalarmv.rawptr());
	EXPECT_EQ(compptr, compmv.rawptr());
	EXPECT_TRUE(scalarmv.allocshape_is(sshape));
	EXPECT_TRUE(compmv.allocshape_is(cshape));

	scalarassign = std::move(scalarmv);
	compassign = std::move(compmv);

	EXPECT_FALSE(scalarmv.is_alloc());
	EXPECT_FALSE(compmv.is_alloc());
	EXPECT_EQ(scalarptr, scalarassign.rawptr());
	EXPECT_EQ(compptr, compassign.rawptr());
	EXPECT_TRUE(scalarassign.allocshape_is(sshape));
	EXPECT_TRUE(compassign.allocshape_is(cshape));
}


// cover tensor
// get_shape, n_elems, rank. dims
TEST(TENSOR, Shape_B002)
{
	tensorshape singular(std::vector<size_t>{1});
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_shape();

	mock_tensor undef;
	mock_tensor scalar(FUZZ<double>::get(1)[0]);
	mock_tensor incom(pshape);
	mock_tensor comp(cshape);

	EXPECT_TRUE(tensorshape_equal(undef.get_shape(), {}));
	EXPECT_TRUE(tensorshape_equal(singular, scalar.get_shape()));
	EXPECT_TRUE(tensorshape_equal(pshape, incom.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape, comp.get_shape()));

	EXPECT_EQ(0, undef.n_elems());
	EXPECT_EQ(1, scalar.n_elems());
	EXPECT_EQ(0, incom.n_elems());
	EXPECT_EQ(cshape.n_elems(), comp.n_elems());

	EXPECT_EQ(0, undef.rank());
	EXPECT_EQ(1, scalar.rank());
	EXPECT_EQ(pshape.rank(), incom.rank());
	EXPECT_EQ(cshape.rank(), comp.rank());

	EXPECT_TRUE(undef.dims().empty());
	std::vector<size_t> sv = scalar.dims();
	ASSERT_EQ(1, sv.size());
	EXPECT_EQ(1, sv[0]);

	std::vector<size_t> expects = pshape.as_list();
	std::vector<size_t> expectc = cshape.as_list();
	EXPECT_TRUE(std::equal(expects.begin(), expects.end(), incom.dims().begin()));
	EXPECT_TRUE(std::equal(expectc.begin(), expectc.end(), comp.dims().begin()));
}


// cover tensor
// is_same_size
TEST(TENSOR, IsSameSize_B003)
{
	tensorshape singular(std::vector<size_t>{1});
	tensorshape cshape = random_shape();
	std::vector<size_t> cv = cshape.as_list();
	tensorshape ishape = make_incompatible(cv); // not same as cshape
	mock_tensor bad(ishape);

	mock_tensor undef;
	mock_tensor scalar(FUZZ<double>::get(1)[0]);
	mock_tensor comp(cshape);

	{
		tensorshape pshape = make_partial(cv); // same as cshape
		mock_tensor pcom(pshape);

		// allowed compatible
		// pcom, undef are both unallocated
		EXPECT_FALSE(undef.is_alloc());
		EXPECT_FALSE(pcom.is_alloc());
		// undef is same as anything
		EXPECT_TRUE(undef.is_same_size(bad));
		EXPECT_TRUE(undef.is_same_size(comp));
		EXPECT_TRUE(undef.is_same_size(scalar));
		EXPECT_TRUE(undef.is_same_size(pcom));
		// pcom is same as comp, but not bad or scalar
		EXPECT_TRUE(pcom.is_same_size(comp));
		EXPECT_FALSE(pcom.is_same_size(bad));
		EXPECT_FALSE(pcom.is_same_size(scalar));
	}

	// trimmed compatible
	{
		// padd cv
		std::vector<size_t> npads = FUZZ<size_t>::get(4, {3, 17});
		tensorshape p1 = padd(cv, npads[0], npads[1]); // same
		tensorshape p2 = padd(cv, npads[2], npads[3]); // same
		cv.push_back(2);
		tensorshape p3 = padd(cv, npads[2], npads[3]); // not same
		mock_tensor comp2(p1);
		mock_tensor comp3(p2);
		mock_tensor bad2(p3);

		EXPECT_TRUE(comp2.is_alloc());
		EXPECT_TRUE(comp3.is_alloc());
		EXPECT_TRUE(bad.is_alloc());

		EXPECT_TRUE(comp.is_same_size(comp2));
		EXPECT_TRUE(comp2.is_same_size(comp3));
		EXPECT_TRUE(comp.is_same_size(comp3));

		EXPECT_FALSE(comp.is_same_size(bad));
		EXPECT_FALSE(comp2.is_same_size(bad));
		EXPECT_FALSE(comp3.is_same_size(bad));

		EXPECT_FALSE(comp.is_same_size(bad2));
		EXPECT_FALSE(comp2.is_same_size(bad2));
		EXPECT_FALSE(comp3.is_same_size(bad2));
	}

}


// cover tensor
// is_compatible_with tensor
TEST(TENSOR, IsCompatibleWithTensor_B004)
{
	tensorshape cshape = random_shape();
	std::vector<size_t> cv = cshape.as_list();
	tensorshape ishape = make_incompatible(cv); // not same as cshape
	tensorshape pshape = make_partial(cv); // same as cshape

	mock_tensor undef;
	mock_tensor scalar(FUZZ<double>::get(1)[0]);
	mock_tensor comp(cshape);
	mock_tensor pcom(pshape);
	mock_tensor bad(ishape);

	// undefined tensor is compatible with anything
	EXPECT_TRUE(undef.is_compatible_with(undef));
	EXPECT_TRUE(undef.is_compatible_with(scalar));
	EXPECT_TRUE(undef.is_compatible_with(comp));
	EXPECT_TRUE(undef.is_compatible_with(pcom));
	EXPECT_TRUE(undef.is_compatible_with(bad));

	EXPECT_TRUE(pcom.is_compatible_with(comp));
	EXPECT_TRUE(pcom.is_compatible_with(pcom));
	EXPECT_FALSE(pcom.is_compatible_with(bad));

	EXPECT_FALSE(bad.is_compatible_with(comp));
}


// cover tensor
// is_compatible_with vector
TEST(TENSOR, IsCompatibleWithVector_B005)
{
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_shape();

	mock_tensor undef;
	mock_tensor comp(cshape);
	mock_tensor pcom(pshape);

	std::vector<double> zerodata;
	size_t cp = cshape.n_elems();
	std::vector<double> lowerdata = FUZZ<double>::get(cp-FUZZ<size_t>::get(1, {1, cp-1})[0]);
	std::vector<double> exactdata = FUZZ<double>::get(cp);
	std::vector<double> upperdata = FUZZ<double>::get(cp+FUZZ<size_t>::get(1, {1, cp-1})[0]);

	EXPECT_TRUE(comp.is_compatible_with(exactdata));
	EXPECT_FALSE(comp.is_compatible_with(lowerdata));
	EXPECT_FALSE(comp.is_compatible_with(upperdata));

	EXPECT_TRUE(comp.is_loosely_compatible_with(exactdata));
	EXPECT_TRUE(comp.is_loosely_compatible_with(lowerdata));
	EXPECT_FALSE(comp.is_loosely_compatible_with(upperdata));

	size_t np = pshape.n_known();
	std::vector<double> lowerdata2 = FUZZ<double>::get(np-FUZZ<size_t>::get(1, {1, np-1})[0]);
	std::vector<double> exactdata2 = FUZZ<double>::get(np);
	size_t mod = np*FUZZ<size_t>::get(1, {2, 15})[0];
	std::vector<double> moddata = FUZZ<double>::get(mod);
	std::vector<double> upperdata2 = FUZZ<double>::get(mod+1);

	EXPECT_TRUE(pcom.is_compatible_with(exactdata2));
	EXPECT_TRUE(pcom.is_compatible_with(moddata));
	EXPECT_FALSE(pcom.is_compatible_with(lowerdata2));
	EXPECT_FALSE(pcom.is_compatible_with(upperdata2));

	EXPECT_TRUE(pcom.is_loosely_compatible_with(exactdata2));
	EXPECT_TRUE(pcom.is_loosely_compatible_with(moddata));
	EXPECT_TRUE(pcom.is_loosely_compatible_with(lowerdata2));
	EXPECT_TRUE(pcom.is_loosely_compatible_with(upperdata2));

	// undef is compatible with everything
	EXPECT_TRUE(undef.is_compatible_with(exactdata));
	EXPECT_TRUE(undef.is_compatible_with(exactdata2));
	EXPECT_TRUE(undef.is_compatible_with(lowerdata));
	EXPECT_TRUE(undef.is_compatible_with(lowerdata2));
	EXPECT_TRUE(undef.is_compatible_with(upperdata));
	EXPECT_TRUE(undef.is_compatible_with(upperdata2));
	EXPECT_TRUE(undef.is_compatible_with(moddata));

	EXPECT_TRUE(undef.is_loosely_compatible_with(exactdata));
	EXPECT_TRUE(undef.is_loosely_compatible_with(exactdata2));
	EXPECT_TRUE(undef.is_loosely_compatible_with(lowerdata));
	EXPECT_TRUE(undef.is_loosely_compatible_with(lowerdata2));
	EXPECT_TRUE(undef.is_loosely_compatible_with(upperdata));
	EXPECT_TRUE(undef.is_loosely_compatible_with(upperdata2));
	EXPECT_TRUE(undef.is_loosely_compatible_with(moddata));
}


// covers tensor
// guess_shape
TEST(TENSOR, GuessShape_B006)
{
}


// cover tensor
// get, expose
TEST(TENSOR, Get_B007)
{
//	size_t x = 2, y = 3, z = 4;
//	
//	const double constant = (double) rand();
//	const_init<double> init(constant);
//	tensor<double> ten(std::vector<size_t>{x, y, z});
//	init(ten);
//	
//	for (size_t i = 0; i < x; i++) {
//	for (size_t j = 0; j < y; j++) {
//	for (size_t k = 0; k < z; k++) {
//	EXPECT_EQ(constant, ten.get({i,j,k}));
//	}
//	}
//	}
}


// cover tensor
// set_shape
TEST(TENSOR, Reshape_B008)
{}


// cover tensor
// default allocate
TEST(TENSOR, Allocate_B009)
{
//	tensor<double> t1(std::vector<size_t>{1, 2, 3});
//	tensor<double> u1(std::vector<size_t>{0, 1, 2});
//	// exactly compatible except, input is undefined
//	EXPECT_DEATH(t1.allocate(std::vector<size_t>{0, 2, 3}), ".*");
//	// Either equivalent shape
//	t1.allocate(std::vector<size_t>{1, 2, 3});
//	// Or none at all
//	t1.allocate();
//	// Undefined tensors absolutely require a fully defined shape on allocation
//	EXPECT_DEATH(u1.allocate(std::vector<size_t>{0, 2, 3}), ".*");
//	EXPECT_DEATH(u1.allocate(), ".*");
//	// same number of elements, but different shape
//	EXPECT_DEATH(t1.allocate(std::vector<size_t>{3, 2, 1}), ".*");
//	// technically the same except different rank
//	EXPECT_DEATH(t1.allocate(std::vector<size_t>{1, 2, 3, 1}), ".*");
}


// cover tensor
// deallocate
TEST(TENSOR, Dealloc_B010)
{}


// cover tensor
// allocate shape
TEST(TENSOR, AllocateShape_B011)
{}


// cover tensor
// copy_from
TEST(TENSOR, CopyWithShape_B012)
{}


#endif /* DISABLE_TENSOR_TEST */
