//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include <algorithm>
#include <numeric>

#include "fuzz.h"
#include "util_test.h"
#include "gtest/gtest.h"


//#define DISABLE_SHAPE_TEST
#ifndef DISABLE_SHAPE_TEST


static void generate_shapes (std::vector<size_t>& pcom, std::vector<size_t>& com)
{
	// get an array of size greater than 1 and do not contain 0
	std::vector<size_t> ds = FUZZ::getInt(FUZZ::getInt(1, {2, 17})[0], {1, 6});
	com = ds;
	// inject a 0 into ds
	size_t idx = FUZZ::getInt(1, {0, ds.size()-1})[0];
	ds.insert(ds.begin() + idx, 0);
	pcom = ds;
}


static void generate_moreshapes (std::vector<size_t>& pcom, std::vector<size_t>& com)
{
	// pcom can have more than 1 zero
	pcom = FUZZ::getInt(FUZZ::getInt(1, {12, 61})[0], {0, 1});
	size_t idx = FUZZ::getInt(1, {0, pcom.size()-1})[0];
	pcom.insert(pcom.begin() + idx, 0);

	// com is not similar to pcom
	com = FUZZ::getInt(FUZZ::getInt(1, {12, 31})[0], {1, 2});
}


// cover tensorshape
// default and vector constructor
// copy constructor and assignment,
// and vector assignment
TEST(TENSORSHAPE, Copy_A000)
{
	FUZZ::delim();
	tensorshape incom_assign;
	tensorshape pcom_assign;
	tensorshape com_assign;

	tensorshape incom_vassign;
	tensorshape pcom_vassign;
	tensorshape com_vassign;

	std::vector<size_t> pds;
	std::vector<size_t> cds;
	generate_shapes(pds, cds);
	// define shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	tensorshape incom_cpy(incom_ts);
	tensorshape pcom_cpy(pcom_ts);
	tensorshape com_cpy(com_ts);

	incom_assign = incom_ts;
	pcom_assign = pcom_ts;
	com_assign = com_ts;

	incom_vassign = std::vector<size_t>{};
	pcom_vassign = pds;
	com_vassign = cds;

	EXPECT_TRUE(tensorshape_equal(incom_cpy, incom_ts));
	EXPECT_TRUE(tensorshape_equal(pcom_cpy, pcom_ts));
	EXPECT_TRUE(tensorshape_equal(com_cpy, com_ts));
	EXPECT_TRUE(tensorshape_equal(incom_assign, incom_ts));
	EXPECT_TRUE(tensorshape_equal(pcom_assign, pcom_ts));
	EXPECT_TRUE(tensorshape_equal(com_assign, com_ts));
	EXPECT_TRUE(tensorshape_equal(incom_vassign, incom_ts));
	EXPECT_TRUE(tensorshape_equal(pcom_vassign, pcom_ts));
	EXPECT_TRUE(tensorshape_equal(com_vassign, com_ts));

	EXPECT_TRUE(tensorshape_equal(pcom_cpy, pds));
	EXPECT_TRUE(tensorshape_equal(com_cpy, cds));
	EXPECT_TRUE(tensorshape_equal(pcom_assign, pds));
	EXPECT_TRUE(tensorshape_equal(com_assign, cds));
	EXPECT_TRUE(tensorshape_equal(pcom_vassign, pds));
	ASSERT_TRUE(tensorshape_equal(com_vassign, cds));
}


// cover tensorshape
// default and vector constructor
// move constructor and assignment
TEST(TENSORSHAPE, Move_A000)
{
	FUZZ::delim();
	tensorshape pcom_assign;
	tensorshape com_assign;
	std::vector<size_t> empty;
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	generate_shapes(pds, cds);
	// define shapes
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	tensorshape pcom_mv(std::move(pcom_ts));
	tensorshape com_mv(std::move(com_ts));

	EXPECT_TRUE(tensorshape_equal(pcom_mv, pds));
	EXPECT_TRUE(tensorshape_equal(com_mv, cds));
	EXPECT_TRUE(tensorshape_equal(pcom_ts, empty));
	EXPECT_TRUE(tensorshape_equal(com_ts, empty));

	pcom_assign = std::move(pcom_mv);
	com_assign = std::move(com_mv);

	EXPECT_TRUE(tensorshape_equal(pcom_assign, pds));
	EXPECT_TRUE(tensorshape_equal(com_assign, cds));
	EXPECT_TRUE(tensorshape_equal(pcom_mv, empty));
	ASSERT_TRUE(tensorshape_equal(com_mv, empty));
}


// covers tensorshape as_list
TEST(TENSORSHAPE, AsList_A001)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	generate_shapes(pds, cds);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	std::vector<size_t> pres = pcom_ts.as_list();
	std::vector<size_t> cres = com_ts.as_list();

	ASSERT_TRUE(incom_ts.as_list().empty());
	ASSERT_TRUE(std::equal(pds.begin(), pds.end(), pres.begin()));
	ASSERT_TRUE(std::equal(cds.begin(), cds.end(), cres.begin()));
}


// covers tensorshape n_elems
TEST(TENSORSHAPE, N_A002)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	generate_moreshapes(pds, cds);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	size_t expect_nelems = 1;
	for (size_t c : cds)
	{
		expect_nelems *= c;
	}

	size_t expect_nknown = 1;
	for (size_t p : pds)
	{
		if (p != 0)
		{
			expect_nknown *= p;
		}
	}

	EXPECT_EQ(0, incom_ts.n_elems());
	EXPECT_EQ(0, pcom_ts.n_elems());
	EXPECT_EQ(expect_nelems, com_ts.n_elems());

	EXPECT_EQ(0, incom_ts.n_known());
	EXPECT_EQ(expect_nknown, pcom_ts.n_known());
	EXPECT_EQ(expect_nelems, com_ts.n_known());
}


// covers tensorshape rank
TEST(TENSORSHAPE, Rank_A003)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	generate_moreshapes(pds, cds);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	EXPECT_EQ(0, incom_ts.rank());
	EXPECT_EQ(pds.size(), pcom_ts.rank());
	ASSERT_EQ(cds.size(), com_ts.rank());
}


// behavior A000
// covers is_compatible_with
TEST(TENSORSHAPE, Compatible_A004)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	generate_shapes(pds, cds);
	generate_moreshapes(pds2, cds2);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);
	tensorshape pcom2_ts(pds2);
	tensorshape com2_ts(cds2);

	for (tensorshape* shape : {&incom_ts, &pcom_ts, &com_ts, &pcom2_ts, &com2_ts})
	{
		EXPECT_TRUE(incom_ts.is_compatible_with(*shape));
	}

	// partially defined are compatible with itself and any value in its zeros
	std::vector<size_t> cds_cpy = cds;
	std::vector<size_t> cds2_cpy = cds2;
	std::vector<size_t> cds_cpy2 = cds;
	std::vector<size_t> cds2_cpy2 = cds2;
	std::vector<size_t> brank = cds2;
	brank.push_back(0);
	size_t idx1 = FUZZ::getInt(1, {0, cds_cpy.size()-1})[0];
	size_t idx2 = FUZZ::getInt(1, {0, cds2_cpy.size()-1})[0];
	cds_cpy[idx1] = 0;
	cds2_cpy[idx2] = 0;
	// ensure cpy2 increments are not made to indices where cpy set to 0
	// this ensure cpy2s are never compatible with cpy
	cds_cpy2[(idx1 + 1) % cds_cpy.size()]++;
	cds2_cpy2[(idx2 + 1) % cds2_cpy.size()]++;
	tensorshape fake_ps(cds_cpy);
	tensorshape fake_ps2(cds2_cpy);
	tensorshape bad_ps(cds_cpy2);
	tensorshape bad_ps2(cds2_cpy2);
	tensorshape bad_ps3(brank);

	// guarantees
	EXPECT_TRUE(fake_ps.is_compatible_with(fake_ps));
	EXPECT_TRUE(fake_ps2.is_compatible_with(fake_ps2));
	EXPECT_TRUE(fake_ps.is_compatible_with(com_ts));
	EXPECT_TRUE(fake_ps2.is_compatible_with(com2_ts));
	EXPECT_FALSE(fake_ps.is_compatible_with(bad_ps));
	EXPECT_FALSE(fake_ps2.is_compatible_with(bad_ps2));
	EXPECT_FALSE(fake_ps2.is_compatible_with(bad_ps3));

	// fully defined are not expected to be compatible with bad
	EXPECT_TRUE(com_ts.is_compatible_with(com_ts));
	EXPECT_FALSE(com_ts.is_compatible_with(bad_ps));
	ASSERT_FALSE(com_ts.is_compatible_with(bad_ps2));
}


// covers is_part_defined
TEST(TENSORSHAPE, PartDef_A005)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	generate_shapes(pds, cds);
	generate_moreshapes(pds2, cds2);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);
	tensorshape pcom2_ts(pds2);
	tensorshape com2_ts(cds2);

	ASSERT_FALSE(incom_ts.is_part_defined());
	ASSERT_TRUE(pcom_ts.is_part_defined());
	ASSERT_TRUE(pcom2_ts.is_part_defined());
	ASSERT_TRUE(com_ts.is_part_defined());
	ASSERT_TRUE(com2_ts.is_part_defined());
}


// covers is_fully_defined and assert_is_fully defined
TEST(TENSORSHAPE, FullDef_A006)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	generate_shapes(pds, cds);
	generate_moreshapes(pds2, cds2);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);
	tensorshape pcom2_ts(pds2);
	tensorshape com2_ts(cds2);

	EXPECT_FALSE(incom_ts.is_fully_defined());
	EXPECT_FALSE(pcom_ts.is_fully_defined());
	EXPECT_FALSE(pcom2_ts.is_fully_defined());
	EXPECT_TRUE(com_ts.is_fully_defined());
	EXPECT_TRUE(com2_ts.is_fully_defined());

	com_ts.assert_is_fully_defined();
	com2_ts.assert_is_fully_defined();
	EXPECT_DEATH(pcom_ts.assert_is_fully_defined(), ".*");
	ASSERT_DEATH(pcom2_ts.assert_is_fully_defined(), ".*");
}


// covers assert_has_rank and assert_same_rank
TEST(TENSORSHAPE, RankAssert_A007)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	generate_shapes(pds, cds);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);
	std::vector<size_t> dum = pds;
	dum.pop_back();
	tensorshape dummys(dum); // same rank as cds

	com_ts.assert_has_rank(cds.size());
	pcom_ts.assert_has_rank(pds.size());
	incom_ts.assert_has_rank(FUZZ::getInt(1)[0]);
	EXPECT_DEATH(com_ts.assert_has_rank(cds.size()+1), ".*");
	EXPECT_DEATH(pcom_ts.assert_has_rank(pds.size()+1), ".*");

	com_ts.assert_same_rank(dummys);
	com_ts.assert_same_rank(com_ts);
	pcom_ts.assert_same_rank(pcom_ts);
	incom_ts.assert_same_rank(dummys);
	incom_ts.assert_same_rank(com_ts);
	incom_ts.assert_same_rank(pcom_ts);
	EXPECT_DEATH(pcom_ts.assert_same_rank(dummys), ".*");
	EXPECT_DEATH(pcom_ts.assert_same_rank(com_ts), ".*");
	ASSERT_DEATH(com_ts.assert_same_rank(pcom_ts), ".*");
}


// covers undefine, dependent on is_part_defined
TEST(TENSORSHAPE, Undefine_A008)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	generate_shapes(pds, cds);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	EXPECT_FALSE(incom_ts.is_part_defined());
	EXPECT_TRUE(pcom_ts.is_part_defined());
	EXPECT_TRUE(com_ts.is_part_defined());

	incom_ts.undefine();
	pcom_ts.undefine();
	com_ts.undefine();

	EXPECT_FALSE(incom_ts.is_part_defined());
	EXPECT_FALSE(pcom_ts.is_part_defined());
	ASSERT_FALSE(com_ts.is_part_defined());
}


// covers merge_with
TEST(TENSORSHAPE, Merge_A009)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	generate_shapes(pds, cds);
	generate_moreshapes(pds2, cds2);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);
	tensorshape pcom2_ts(pds2);
	tensorshape com2_ts(cds2);

	// incomplete shape can merge with anything
	for (tensorshape* shape : {&pcom_ts, &com_ts, &pcom2_ts, &com2_ts, &incom_ts})
	{
		tensorshape merged = incom_ts.merge_with(*shape);
		// we're expecting merged shape to be the same as input shape
		EXPECT_TRUE(tensorshape_equal(merged, *shape));
	}

	// partially defined merging with
	// fully defined yields fully defined
	// incompatible merging favors the calling member
	std::vector<size_t> cds_cpy = cds;
	std::vector<size_t> cds2_cpy = cds2;
	std::vector<size_t> cds_cpy2 = cds;
	std::vector<size_t> cds2_cpy2 = cds2;
	size_t idx1 = FUZZ::getInt(1, {0, cds_cpy.size()-1})[0];
	size_t idx2 = FUZZ::getInt(1, {0, cds2_cpy.size()-1})[0];
	cds_cpy[idx1] = 0;
	cds2_cpy[idx2] = 0;
	// ensure cpy2 increments are not made to indices where cpy set to 0
	// this ensure cpy2s are never compatible with cpy
	cds_cpy2[(idx1 + 1) % cds_cpy.size()]++;
	cds2_cpy2[(idx2 + 1) % cds2_cpy.size()]++;
	tensorshape fake_ps(cds_cpy);
	tensorshape fake_ps2(cds2_cpy);
	tensorshape incompatible(cds_cpy2);
	tensorshape incompatible2(cds2_cpy2);

	EXPECT_TRUE(tensorshape_equal(fake_ps.merge_with(com_ts), com_ts));
	EXPECT_TRUE(tensorshape_equal(fake_ps2.merge_with(com2_ts), com2_ts));
	EXPECT_TRUE(tensorshape_equal(
		incompatible.merge_with(com_ts), incompatible));
	EXPECT_TRUE(tensorshape_equal(
		incompatible2.merge_with(com2_ts), incompatible2));

	// merging different ranks will error
	assert(pds.size() > cds.size()); // true by generation implementation
	EXPECT_THROW(pcom_ts.merge_with(com_ts), std::logic_error);
}


// covers trim, dependent on rank
TEST(TENSORSHAPE, Trim_A010)
{
	FUZZ::delim();
	std::vector<size_t> ids;
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	generate_moreshapes(pds, cds);
	// padd a bunch of ones to pds and cds
	std::vector<size_t> npads = FUZZ::getInt(5, {3, 12});
	ids.insert(ids.begin(), npads[0], 1);
	std::vector<size_t> fakepds(npads[1], 1);
	std::vector<size_t> fakecds(npads[2], 1);
	fakepds.push_back(2); // ensures trimming never proceeds inward
	fakepds.insert(fakepds.end(), pds.begin(), pds.end());
	fakepds.push_back(2); // ensures trimming never proceeds inward
	fakepds.insert(fakepds.end(), npads[3], 1);
	fakecds.push_back(2); // ensures trimming never proceeds inward
	fakecds.insert(fakecds.end(), cds.begin(), cds.end());
	fakecds.push_back(2); // ensures trimming never proceeds inward
	fakecds.insert(fakecds.end(), npads[4], 1);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape fakeincom_ts(ids);
	tensorshape pcom_ts(fakepds);
	tensorshape com_ts(fakecds);

	EXPECT_EQ(0, incom_ts.trim().rank());
	EXPECT_LT(0, fakeincom_ts.rank());
	EXPECT_EQ(0, fakeincom_ts.trim().rank());

	EXPECT_EQ(pds.size()+2, pcom_ts.trim().rank());
	ASSERT_EQ(cds.size()+2, com_ts.trim().rank());
}


// covers concatenate
TEST(TENSORSHAPE, Concat_A011)
{
	FUZZ::delim();
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	generate_moreshapes(pds, cds);
	// define partial and complete shapes
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	// undefined concatenating anything is that thing
	tensorshape none1 = incom_ts.concatenate(com_ts);
	tensorshape none2 = com_ts.concatenate(incom_ts);
	tensorshape none3 = incom_ts.concatenate(pcom_ts);
	tensorshape none4 = pcom_ts.concatenate(incom_ts);
	EXPECT_TRUE(tensorshape_equal(none1, com_ts));
	EXPECT_TRUE(tensorshape_equal(none2, com_ts));
	EXPECT_TRUE(tensorshape_equal(none3, pcom_ts));
	EXPECT_TRUE(tensorshape_equal(none4, pcom_ts));

	std::vector<size_t> straight = com_ts.concatenate(pcom_ts).as_list();
	std::vector<size_t> backcat = pcom_ts.concatenate(com_ts).as_list();

	ASSERT_TRUE(straight.size() == backcat.size());
	ASSERT_TRUE(straight.size() == cds.size() + pds.size());
	std::vector<size_t> expect_str8 = cds;
	std::vector<size_t> expect_revr = pds;
	expect_str8.insert(expect_str8.end(), pds.begin(), pds.end());
	expect_revr.insert(expect_revr.end(), cds.begin(), cds.end());
	EXPECT_TRUE(std::equal(straight.begin(),
		straight.end(), expect_str8.begin()));
	EXPECT_TRUE(std::equal(backcat.begin(),
		backcat.end(), expect_revr.begin()));
}


// covers with_rank, with_rank_at_least, with_rank_at_most, depends on rank
TEST(TENSORSHAPE, WithRank_A012)
{
	FUZZ::delim();
	std::vector<size_t> ids;
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	generate_moreshapes(pds, cds);
	tensorshape incom_ts;
	tensorshape pcom_ts(pds);
	tensorshape com_ts(cds);

	// expand rank
	size_t peak = std::max(pds.size(), cds.size());
	size_t trough = std::min(pds.size(), cds.size());
	std::vector<size_t> bounds = FUZZ::getInt(2, {3, trough});
	size_t upperbound = peak + bounds[0];
	size_t lowerbound = trough - bounds[1];
	// expansion
	EXPECT_EQ(upperbound, incom_ts.with_rank(upperbound).rank());
	EXPECT_EQ(upperbound, pcom_ts.with_rank(upperbound).rank());
	EXPECT_EQ(upperbound, com_ts.with_rank(upperbound).rank());
	// compression
	EXPECT_EQ(lowerbound, incom_ts.with_rank(lowerbound).rank());
	EXPECT_EQ(lowerbound, pcom_ts.with_rank(lowerbound).rank());
	EXPECT_EQ(lowerbound, com_ts.with_rank(lowerbound).rank());

	// favor higher dimensionalities
	EXPECT_EQ(upperbound, incom_ts.with_rank_at_least(upperbound).rank());
	EXPECT_EQ(upperbound, pcom_ts.with_rank_at_least(upperbound).rank());
	EXPECT_EQ(upperbound, com_ts.with_rank_at_least(upperbound).rank());
	EXPECT_EQ(lowerbound, incom_ts.with_rank_at_least(lowerbound).rank());
	EXPECT_EQ(pds.size(), pcom_ts.with_rank_at_least(lowerbound).rank());
	EXPECT_EQ(cds.size(), com_ts.with_rank_at_least(lowerbound).rank());

	// favor lower dimensionalities
	EXPECT_EQ(0, incom_ts.with_rank_at_most(upperbound).rank());
	EXPECT_EQ(pds.size(), pcom_ts.with_rank_at_most(upperbound).rank());
	EXPECT_EQ(cds.size(), com_ts.with_rank_at_most(upperbound).rank());
	EXPECT_EQ(0, incom_ts.with_rank_at_most(lowerbound).rank());
	EXPECT_EQ(lowerbound, pcom_ts.with_rank_at_most(lowerbound).rank());
	EXPECT_EQ(lowerbound, com_ts.with_rank_at_most(lowerbound).rank());
}


// TENSORSHAPE TO PROTOCOL BUFFER
TEST(TENSORSHAPE, Proto) {}


#endif /* DISABLE_SHAPE_TEST */

#endif /* DISABLE_TENSOR_MODULE_TESTS */
