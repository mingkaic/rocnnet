//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"

#include "mocks/mock_tensor.h"


#ifndef DISABLE_TENSOR_TEST


static tensorshape random_partialshape (void)
{
	std::vector<size_t> rlist = random_def_shape().as_list();
	size_t nzeros = FUZZ::getInt(1, "nzeros", {1, 5})[0];
	for (size_t i = 0; i < nzeros; i++)
	{
		size_t zidx = FUZZ::getInt(1, "zidx", {0, rlist.size()})[0];
		rlist.insert(rlist.begin()+zidx, 0);
	}
	return tensorshape(rlist);
}


// cover scalar tensor constructor
TEST(TENSOR, ScalarConstructor_B000)
{
	FUZZ::reset_logger();
	std::vector<double> vals = FUZZ::getDouble(3, "vals");
	double value = vals[0];
	mock_tensor scalar(value);
	EXPECT_TRUE(scalar.clean());
	EXPECT_TRUE(scalar.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar.total_bytes());
	EXPECT_EQ(value, *scalar.rawptr());

	value = vals[1];
	mock_tensor scalar2(value);
	EXPECT_TRUE(scalar2.clean());
	EXPECT_TRUE(scalar2.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar2.total_bytes());
	EXPECT_EQ(value, *scalar2.rawptr());

	value = vals[2];
	mock_tensor scalar3(value);
	EXPECT_TRUE(scalar3.clean());
	EXPECT_TRUE(scalar3.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar3.total_bytes());
	EXPECT_EQ(value, *scalar3.rawptr());
}


// cover tensor
// default, shape constructors,
// is_alloc, total_bytes
TEST(TENSOR, Construct_B001)
{
	FUZZ::reset_logger();
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();

	mock_tensor undef;
	mock_tensor incom(pshape);
	mock_tensor comp(cshape);

	EXPECT_TRUE(undef.clean());
	EXPECT_TRUE(incom.clean());
	EXPECT_TRUE(comp.clean());

	EXPECT_FALSE(undef.is_alloc());
	EXPECT_FALSE(incom.is_alloc());
	EXPECT_TRUE(comp.is_alloc());

	EXPECT_EQ((size_t) 0, undef.total_bytes());
	EXPECT_EQ((size_t) 0, incom.total_bytes());
	EXPECT_EQ((size_t) sizeof(double) * cshape.n_elems(),
		comp.total_bytes());
}


// cover tensor
// clone and assignment
TEST(TENSOR, Copy_B002)
{
	FUZZ::reset_logger();
	mock_tensor undefassign;
	mock_tensor scalarassign;
	mock_tensor incomassign;
	mock_tensor compassign;

	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();

	mock_tensor undef;
	mock_tensor scalar(FUZZ::getDouble(1, "scalar.data")[0]);
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
TEST(TENSOR, Move_B002)
{
	FUZZ::reset_logger();
	mock_tensor scalarassign;
	mock_tensor compassign;

	tensorshape sshape(std::vector<size_t>{1});
	tensorshape cshape = random_def_shape();
	mock_tensor scalar(FUZZ::getDouble(1, "scalar.data")[0]);
	mock_tensor comp(cshape);

	double* scalarptr = scalar.rawptr();
	double* compptr = comp.rawptr();

	mock_tensor scalarmv(std::move(scalar));
	mock_tensor compmv(std::move(comp));

	EXPECT_TRUE(scalar.clean());
	EXPECT_TRUE(comp.clean());
	EXPECT_TRUE(scalarmv.clean());
	EXPECT_TRUE(compmv.clean());

	EXPECT_FALSE(scalar.is_alloc());
	EXPECT_FALSE(comp.is_alloc());
	EXPECT_EQ(scalarptr, scalarmv.rawptr());
	EXPECT_EQ(compptr, compmv.rawptr());
	EXPECT_TRUE(scalarmv.allocshape_is(sshape));
	EXPECT_TRUE(compmv.allocshape_is(cshape));

	scalarassign = std::move(scalarmv);
	compassign = std::move(compmv);

	EXPECT_TRUE(scalarmv.clean());
	EXPECT_TRUE(compmv.clean());
	EXPECT_TRUE(scalarassign.clean());
	EXPECT_TRUE(compassign.clean());

	EXPECT_FALSE(scalarmv.is_alloc());
	EXPECT_FALSE(compmv.is_alloc());
	EXPECT_EQ(scalarptr, scalarassign.rawptr());
	EXPECT_EQ(compptr, compassign.rawptr());
	EXPECT_TRUE(scalarassign.allocshape_is(sshape));
	EXPECT_TRUE(compassign.allocshape_is(cshape));
}


// cover tensor
// get_shape, n_elems, rank. dims
TEST(TENSOR, Shape_B003)
{
	FUZZ::reset_logger();
	tensorshape singular(std::vector<size_t>{1});
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();

	mock_tensor undef;
	mock_tensor scalar(FUZZ::getDouble(1, "scalar.data")[0]);
	mock_tensor incom(pshape);
	mock_tensor comp(cshape);

	EXPECT_TRUE(tensorshape_equal(undef.get_shape(), {}));
	EXPECT_TRUE(tensorshape_equal(singular, scalar.get_shape()));
	EXPECT_TRUE(tensorshape_equal(pshape, incom.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape, comp.get_shape()));

	EXPECT_EQ((size_t) 0, undef.n_elems());
	EXPECT_EQ((size_t) 1, scalar.n_elems());
	EXPECT_EQ((size_t) 0, incom.n_elems());
	EXPECT_EQ(cshape.n_elems(), comp.n_elems());

	EXPECT_EQ((size_t) 0, undef.rank());
	EXPECT_EQ((size_t) 1, scalar.rank());
	EXPECT_EQ(pshape.rank(), incom.rank());
	EXPECT_EQ(cshape.rank(), comp.rank());

	EXPECT_TRUE(undef.dims().empty());
	std::vector<size_t> sv = scalar.dims();
	ASSERT_EQ((size_t) 1, sv.size());
	EXPECT_EQ((size_t) 1, sv[0]);

	std::vector<size_t> expects = pshape.as_list();
	std::vector<size_t> expectc = cshape.as_list();
	EXPECT_TRUE(std::equal(expects.begin(), expects.end(), incom.dims().begin()));
	EXPECT_TRUE(std::equal(expectc.begin(), expectc.end(), comp.dims().begin()));
}


// cover tensor
// is_same_size
TEST(TENSOR, IsSameSize_B004)
{
	FUZZ::reset_logger();
	tensorshape cshape = random_def_shape();
	std::vector<size_t> cv = cshape.as_list();
	tensorshape ishape = make_incompatible(cv); // not same as cshape
	mock_tensor bad(ishape);
	mock_tensor undef;
	mock_tensor scalar(FUZZ::getDouble(1, "scalar.data")[0]);
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
		std::vector<size_t> npads = FUZZ::getInt(4, "npads", {3, 17});
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
TEST(TENSOR, IsCompatibleWithTensor_B005)
{
	FUZZ::reset_logger();
	tensorshape cshape = random_def_shape();
	std::vector<size_t> cv = cshape.as_list();
	tensorshape ishape = make_incompatible(cv); // not same as cshape
	tensorshape pshape = make_partial(cv); // same as cshape
	mock_tensor undef;
	mock_tensor scalar(FUZZ::getDouble(1, "scalar.data")[0]);
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
TEST(TENSOR, IsCompatibleWithVector_B006)
{
	FUZZ::reset_logger();
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();

	mock_tensor undef;
	mock_tensor comp(cshape);
	mock_tensor pcom(pshape);

	std::vector<double> zerodata;
	size_t cp = cshape.n_elems();
	std::vector<double> lowerdata;
	if (cp < 3)
	{
		lowerdata = FUZZ::getDouble(1, "lowerdata if cp < 3");
	}
	else
	{
		lowerdata = FUZZ::getDouble(cp-FUZZ::getInt(1, "cp-this", {1, cp-1})[0], "lowerdata if cp >= 3");
	}
	std::vector<double> exactdata = FUZZ::getDouble(cp, "exact data");
	std::vector<double> upperdata = FUZZ::getDouble(cp+FUZZ::getInt(1, "cp+this", {1, cp-1})[0], "upperdata");

	EXPECT_TRUE(comp.is_compatible_with(exactdata));
	EXPECT_FALSE(comp.is_compatible_with(lowerdata));
	EXPECT_FALSE(comp.is_compatible_with(upperdata));

	EXPECT_TRUE(comp.is_loosely_compatible_with(exactdata));
	EXPECT_TRUE(comp.is_loosely_compatible_with(lowerdata));
	EXPECT_FALSE(comp.is_loosely_compatible_with(upperdata));

	size_t np = pshape.n_known();
	std::vector<double> lowerdata2;
	if (np < 3)
	{
		lowerdata2 = FUZZ::getDouble(1, "lowerdata2 if np < 3");
	}
	else
	{
		lowerdata2 = FUZZ::getDouble(np-FUZZ::getInt(1, "np-this", {1, np-1})[0], "lowerdata2 if np >= 3");
	}
	std::vector<double> exactdata2 = FUZZ::getDouble(np, "exactdata2");
	size_t mod = np*FUZZ::getInt(1, "mod=np*this", {2, 15})[0];
	std::vector<double> moddata = FUZZ::getDouble(mod, "moddata");
	std::vector<double> upperdata2 = FUZZ::getDouble(mod+1, "upperdata2");

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
TEST(TENSOR, GuessShape_B007)
{
	FUZZ::reset_logger();
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();
	mock_tensor undef;
	mock_tensor comp(cshape);
	mock_tensor pcom(pshape);

	std::vector<double> zerodata;
	size_t cp = cshape.n_elems();
	std::vector<double> lowerdata;
	if (cp < 3)
	{
		lowerdata = FUZZ::getDouble(1, "lowerdata if cp < 3");
	}
	else
	{
		lowerdata = FUZZ::getDouble(cp-FUZZ::getInt(1, "cp-this", {1, cp-1})[0], "lowerdata if cp >= 3");
	}
	std::vector<double> exactdata = FUZZ::getDouble(cp, "exactdata");
	std::vector<double> upperdata = FUZZ::getDouble(cp+FUZZ::getInt(1, "cp+this", {1, cp-1})[0], "upperdata");

	// allowed are fully defined
	optional<tensorshape> cres = comp.guess_shape(exactdata);
	ASSERT_TRUE((bool)cres);
	EXPECT_TRUE(tensorshape_equal(cshape, *cres));
	EXPECT_FALSE((bool)comp.guess_shape(lowerdata));
	EXPECT_FALSE((bool)comp.guess_shape(upperdata));

	size_t np = pshape.n_known();
	std::vector<double> lowerdata2;
	if (np < 3)
	{
		lowerdata2 = FUZZ::getDouble(1, "lowerdata2 if np < 3");
	}
	else
	{
		lowerdata2 = FUZZ::getDouble(np-FUZZ::getInt(1, "np-this", {1, np-1})[0], "lowerdata2 if np >= 3");
	}
	std::vector<double> exactdata2 = FUZZ::getDouble(np, "exactdata2");
	size_t mod = np*FUZZ::getInt(1, "mod=np*this", {2, 15})[0];
	std::vector<double> moddata = FUZZ::getDouble(mod, "moddata");
	std::vector<double> upperdata2 = FUZZ::getDouble(mod+1, "upperdata2");

	std::vector<size_t> pv = pshape.as_list();
	size_t unknown = pv.size();
	for (size_t i = 0; i < pv.size(); i++)
	{
		if (0 == pv[i])
		{
			if (unknown > i)
			{
				unknown = i;
			}
			pv[i] = 1;
		}
	}
	std::vector<size_t> pv2 = pv;
	pv2[unknown] = ceil((double) moddata.size() / (double) np);
	// allowed are partially defined
	optional<tensorshape> pres = pcom.guess_shape(exactdata2);
	optional<tensorshape> pres2 = pcom.guess_shape(moddata);
	ASSERT_TRUE((bool)pres);
	ASSERT_TRUE((bool)pres2);
	EXPECT_TRUE(tensorshape_equal(*pres, pv));
	EXPECT_TRUE(tensorshape_equal(*pres2, pv2));
	EXPECT_FALSE((bool)pcom.guess_shape(lowerdata2));
	EXPECT_FALSE((bool)pcom.guess_shape(upperdata2));

	// allowed are undefined
	optional<tensorshape> ures = undef.guess_shape(exactdata);
	optional<tensorshape> ures2 = undef.guess_shape(exactdata2);
	optional<tensorshape> ures3 = undef.guess_shape(lowerdata);
	optional<tensorshape> ures4 = undef.guess_shape(lowerdata2);
	optional<tensorshape> ures5 = undef.guess_shape(upperdata);
	optional<tensorshape> ures6 = undef.guess_shape(upperdata2);
	optional<tensorshape> ures7 = undef.guess_shape(moddata);
	ASSERT_TRUE((bool)ures);
	ASSERT_TRUE((bool)ures2);
	ASSERT_TRUE((bool)ures3);
	ASSERT_TRUE((bool)ures4);
	ASSERT_TRUE((bool)ures5);
	ASSERT_TRUE((bool)ures6);
	ASSERT_TRUE((bool)ures7);
	EXPECT_TRUE(tensorshape_equal(*ures, std::vector<size_t>({exactdata.size()})));
	EXPECT_TRUE(tensorshape_equal(*ures2, std::vector<size_t>({exactdata2.size()})));
	EXPECT_TRUE(tensorshape_equal(*ures3, std::vector<size_t>({lowerdata.size()})));
	EXPECT_TRUE(tensorshape_equal(*ures4, std::vector<size_t>({lowerdata2.size()})));
	EXPECT_TRUE(tensorshape_equal(*ures5, std::vector<size_t>({upperdata.size()})));
	EXPECT_TRUE(tensorshape_equal(*ures6, std::vector<size_t>({upperdata2.size()})));
	EXPECT_TRUE(tensorshape_equal(*ures7, std::vector<size_t>({moddata.size()})));
}


// cover tensor
// get, expose
TEST(TENSOR, Get_B008)
{
	FUZZ::reset_logger();
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();
	size_t crank = cshape.rank();
	size_t celem = cshape.n_elems();

	mock_tensor undef;
	mock_tensor pcom(pshape);
	mock_tensor comp(cshape);

	std::vector<double> cv = comp.expose(); // shouldn't die or throw
	// EXPECT_DEATH(undef.expose(), ".*");
	// EXPECT_DEATH(pcom.expose(), ".*");

	size_t pncoord = 1;
	if (crank > 2)
	{
		pncoord = FUZZ::getInt(1, "pncoord if crank > 3", {crank/2, crank-1})[0];
	}
	size_t cncoord = crank;
	size_t rncoord = FUZZ::getInt(1, "rncoord", {15, 127})[0];
	// c coordinates have rank exactly fitting cshape
	// p coordinates have rank less than rank of cshape
	// r coordinates are random coordinates
	std::vector<size_t> ccoord = FUZZ::getInt(cncoord, "ccoord");
	std::vector<size_t> pcoord = FUZZ::getInt(pncoord, "pcoord");
	std::vector<size_t> rcoord = FUZZ::getInt(rncoord, "rcoord");
	EXPECT_THROW(undef.get(pcoord), std::out_of_range);
	EXPECT_THROW(pcom.get(pcoord), std::out_of_range);
	EXPECT_THROW(undef.get(ccoord), std::out_of_range);
	EXPECT_THROW(pcom.get(ccoord), std::out_of_range);
	EXPECT_THROW(undef.get(rcoord), std::out_of_range);
	EXPECT_THROW(pcom.get(rcoord), std::out_of_range);

	std::vector<size_t> cs = cshape.as_list();
	size_t pcoordmax = 0, ccoordmax = 0, rcoordmax = 0;
	for (size_t i = 0, multiplier = 1, cn = cs.size(); i < cn; i++)
	{
		if (i < pncoord)
		{
			pcoordmax += pcoord[i] * multiplier;
		}
		if (i < rncoord)
		{
			rcoordmax += rcoord[i] * multiplier;
		}
		ccoordmax += ccoord[i] * multiplier;
		multiplier *= cs[i];
	}
	
	ASSERT_GT(celem, (size_t) 0);
	if (celem <= pcoordmax)
	{
		EXPECT_THROW(comp.get(pcoord), std::out_of_range);
	}
	else
	{
		ASSERT_GT(cv.size(), pcoordmax);
		EXPECT_EQ(cv[pcoordmax], comp.get(pcoord));
	}
	if (celem <= ccoordmax)
	{
		EXPECT_THROW(comp.get(ccoord), std::out_of_range);
	}
	else
	{
		ASSERT_GT(cv.size(), ccoordmax);
		EXPECT_EQ(cv[ccoordmax], comp.get(ccoord));
	}
	if (celem <= rcoordmax)
	{
		EXPECT_THROW(comp.get(rcoord), std::out_of_range);
	}
	else
	{
		ASSERT_GT(cv.size(), rcoordmax);
		EXPECT_EQ(cv[rcoordmax], comp.get(rcoord));
	}
}


// cover tensor
// set_shape, allocate shape
TEST(TENSOR, Reshape_B009)
{
	FUZZ::reset_logger();
	tensorshape pshape = random_partialshape();
	// make cshape a 2d shape to make testing easy
	// todo: improve to test higher dimensionality
	tensorshape cshape = FUZZ::getInt(2, "cshape", {11, 127});
	std::vector<size_t> cv = cshape.as_list();
	size_t cols = cv[0];
	size_t rows = cv[1];

	mock_tensor undef;
	mock_tensor undef2;
	mock_tensor pcom(pshape);
	mock_tensor comp(cshape);
	mock_tensor comp2(cshape);
	mock_tensor comp3(cshape);
	mock_tensor comp4(cshape);

	// undefined/part defined shape change
	undef.set_shape(pshape);
	EXPECT_TRUE(tensorshape_equal(undef.get_shape(), pshape));
	EXPECT_FALSE(undef.is_alloc());

	undef2.set_shape(cshape);
	pcom.set_shape(cshape);
	EXPECT_TRUE(tensorshape_equal(undef2.get_shape(), cshape));
	EXPECT_TRUE(tensorshape_equal(pcom.get_shape(), cshape));
	EXPECT_FALSE(undef2.is_alloc());
	EXPECT_FALSE(pcom.is_alloc());

	ASSERT_TRUE(comp.is_alloc());
	ASSERT_TRUE(comp2.is_alloc());
	ASSERT_TRUE(comp3.is_alloc());
	ASSERT_TRUE(comp4.is_alloc());
	// comps must be alloc otherwise doubleDArr will fail assertion it != et
	// meaning data size < cv.n_elems
	std::vector<std::vector<double> > ac1 = doubleDArr(comp.expose(), cv);
	std::vector<std::vector<double> > ac2 = doubleDArr(comp2.expose(), cv);
	std::vector<std::vector<double> > ac3 = doubleDArr(comp3.expose(), cv);
	std::vector<std::vector<double> > ac4 = doubleDArr(comp4.expose(), cv);
	// data expansion
	std::vector<size_t> cvexp = cv;
	cvexp[0]++;
	cvexp[1]++;
	comp.set_shape(cvexp);
	std::vector<std::vector<double> > resc1 = doubleDArr(comp.expose(), cvexp);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			EXPECT_EQ(ac1[i][j], resc1[i][j]);
		}
		// check the padding
		EXPECT_EQ((size_t) 0, resc1[i][cols]);
	}
	// check the padding
	for (size_t i = 0; i < cols+1; i++)
	{
		EXPECT_EQ((size_t) 0, resc1[rows][i]);
	}

	// data clipping
	std::vector<size_t> cvcli = cv;
	cvcli[0]--;
	cvcli[1]--;
	comp2.set_shape(cvcli);
	std::vector<std::vector<double> > resc2 = doubleDArr(comp2.expose(), cvcli);
	for (size_t i = 0; i < rows-1; i++)
	{
		for (size_t j = 0; j < cols-1; j++)
		{
			EXPECT_EQ(ac2[i][j], resc2[i][j]);
		}
	}

	// clip in one dimension, expand in another
	std::vector<size_t> cvexpcli = cv;
	std::vector<size_t> cvexpcli2 = cv;
	cvexpcli[0]++;
	cvexpcli[1]--;
	cvexpcli2[0]--;
	cvexpcli2[1]++;
	comp3.set_shape(cvexpcli);
	comp4.set_shape(cvexpcli2);
	std::vector<std::vector<double> > resc3 = doubleDArr(comp3.expose(), cvexpcli);
	std::vector<std::vector<double> > resc4 = doubleDArr(comp4.expose(), cvexpcli2);
	for (size_t i = 0; i < rows-1; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			EXPECT_EQ(ac3[i][j], resc3[i][j]);
		}
		// check the padding
		EXPECT_EQ((size_t) 0, resc3[i][cols]);
	}
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols-1; j++)
		{
			EXPECT_EQ(ac4[i][j], resc4[i][j]);
		}
	}
	// check the padding
	for (size_t i = 0; i < cols-1; i++)
	{
		EXPECT_EQ((size_t) 0, resc4[rows][i]);
	}

	double* p = comp.rawptr();
	double* p2 = comp2.rawptr();
	tensorshape incmpshape = make_incompatible(comp.get_shape().as_list());
	comp.set_shape(incmpshape);
	comp2.set_shape(std::vector<size_t>{});
	ASSERT_NE(p, comp.rawptr());
	ASSERT_EQ(p2, comp2.rawptr());
}


// cover tensor
// default allocate, dependent on set shape
TEST(TENSOR, Allocate_B010)
{
	FUZZ::reset_logger();
	tensorshape cshape = random_def_shape();
	tensorshape pshape = random_partialshape();

	mock_tensor undef;
	mock_tensor pcom(pshape);
	mock_tensor comp(cshape);
	ASSERT_TRUE(comp.is_alloc());
	ASSERT_FALSE(pcom.is_alloc());
	ASSERT_FALSE(undef.is_alloc());
	double* orig = comp.rawptr(); // check to see if comp.rawptr changes later
	EXPECT_FALSE(undef.allocate());
	EXPECT_FALSE(pcom.allocate());
	EXPECT_FALSE(comp.allocate());
	// change allowed shape to defined shape, cshape
	undef.set_shape(cshape);
	pcom.set_shape(cshape);
	EXPECT_TRUE(undef.allocate());
	EXPECT_TRUE(pcom.allocate());
	EXPECT_EQ(orig, comp.rawptr());
}


// cover tensor
// deallocate
TEST(TENSOR, Dealloc_B011)
{
	FUZZ::reset_logger();
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();

	mock_tensor undef;
	mock_tensor pcom(pshape);
	mock_tensor comp(cshape);

	EXPECT_FALSE(undef.is_alloc());
	EXPECT_FALSE(pcom.is_alloc());
	EXPECT_TRUE(comp.is_alloc());
	EXPECT_FALSE(undef.deallocate());
	EXPECT_FALSE(pcom.deallocate());
	EXPECT_TRUE(comp.deallocate());
	EXPECT_FALSE(comp.is_alloc());
}


// cover tensor
// allocate shape
TEST(TENSOR, AllocateShape_B012)
{
	FUZZ::reset_logger();
	tensorshape cshape = random_def_shape();
	std::vector<size_t> cv = cshape.as_list();
	tensorshape cshape2 = make_incompatible(cv);
	tensorshape pshape = make_partial(cv);
	tensorshape pshape2 = make_full_incomp(pshape.as_list(), cv);

	mock_tensor undef;
	mock_tensor pcom(pshape);
	mock_tensor comp(cshape);
	double* orig = comp.rawptr();

	EXPECT_FALSE(undef.allocate(pshape));
	EXPECT_TRUE(undef.allocate(cshape));
	EXPECT_FALSE(pcom.allocate(cshape2));
	EXPECT_TRUE(pcom.allocate(cshape));
	EXPECT_FALSE(comp.allocate(cshape));
	EXPECT_FALSE(comp.allocate(cshape2));
	EXPECT_FALSE(comp.allocate(pshape));
	EXPECT_EQ(orig, comp.rawptr());

	EXPECT_TRUE(tensorshape_equal(cshape, undef.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape, pcom.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape, comp.get_shape()));

	// they're all allocated now
	EXPECT_TRUE(undef.is_alloc());
	EXPECT_TRUE(pcom.is_alloc());
	EXPECT_TRUE(comp.is_alloc());

	ASSERT_TRUE(pcom.allocate(pshape2));
	EXPECT_NE(orig, pcom.rawptr());
}


// cover tensor
// copy_from
TEST(TENSOR, CopyWithShape_B013)
{
	FUZZ::reset_logger();
	tensorshape pshape = random_partialshape();
	tensorshape cshape = random_def_shape();
	tensorshape cshape2 = random_def_shape();
	tensorshape cshape3 = random_def_shape();

	size_t n1 = cshape.n_elems();
	std::vector<double> rawdata1 = FUZZ::getDouble(n1, "rawdata1");
	size_t n2 = cshape2.n_elems();
	std::vector<double> rawdata2 = FUZZ::getDouble(n2, "rawdata2");
	mock_tensor undef;
	mock_tensor pcom(pshape);
	mock_tensor comp(cshape, rawdata1);
	mock_tensor comp2(cshape2, rawdata2);
	double* orig = comp.rawptr();
	double* orig2 = comp2.rawptr();
	std::vector<double> compdata = comp.expose();
	std::vector<double> compdata2 = comp2.expose();

	// copying from unallocated
	EXPECT_FALSE(pcom.copy_from(undef, cshape));
	EXPECT_FALSE(undef.copy_from(pcom, cshape));
	EXPECT_FALSE(pcom.is_alloc());
	EXPECT_FALSE(undef.is_alloc());

	EXPECT_TRUE(undef.copy_from(comp, cshape3));
	EXPECT_TRUE(pcom.copy_from(comp2, cshape3));

	EXPECT_TRUE(comp.copy_from(comp2, cshape3));
	EXPECT_TRUE(comp2.copy_from(comp2, cshape3)); // copy from self

	// pointers are now different
	EXPECT_NE(orig, comp.rawptr());
	EXPECT_NE(orig2, comp2.rawptr());

	EXPECT_TRUE(tensorshape_equal(cshape3, undef.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape3, pcom.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape3, comp.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape3, comp2.get_shape()));

	std::vector<double> undefdata = undef.expose();
	std::vector<double> pdefdata = pcom.expose();

	std::vector<size_t> c1list = cshape.as_list();
	std::vector<size_t> c2list = cshape2.as_list();
	std::vector<size_t> c3list = cshape3.as_list();

	// undef fitted with comp and cshape3
	for (size_t i = 0, n = cshape.n_elems(); i < n; i++)
	{
		std::vector<size_t> incoord = cshape.coordinate_from_idx(i);
		bool b = true;
		for (size_t j = 0, o = incoord.size(); j < o && b; j++)
		{
			if (j >= c3list.size())
			{
				b = incoord[j] == 0;
			}
			else
			{
				b = incoord[j] < c3list[j];
			}
		}
		if (b)
		{
			size_t outidx = cshape3.sequential_idx(incoord);
			EXPECT_EQ(compdata[i], undefdata[outidx]);
		}
	}
	// pdefdata fitted with comp2 and cshape 3
	for (size_t i = 0, n = cshape2.n_elems(); i < n; i++)
	{
		std::vector<size_t> incoord = cshape2.coordinate_from_idx(i);
		bool b = true;
		for (size_t j = 0, o = incoord.size(); j < o && b; j++)
		{
			if (j >= c3list.size())
			{
				b = incoord[j] == 0;
			}
			else
			{
				b = incoord[j] < c3list[j];
			}
		}
		if (b)
		{
			size_t outidx = cshape3.sequential_idx(incoord);
			EXPECT_EQ(compdata2[i], pdefdata[outidx]);
		}
	}
}


// cover tensor
// from_proto
TEST(TENSOR, DISABLED_FromProto_B014)
{
	// should allocate the tensor
	// data and shape are identical to ad hoc values
}


#endif /* DISABLE_TENSOR_TEST */

#endif /* DISABLE_TENSOR_MODULE_TESTS */
