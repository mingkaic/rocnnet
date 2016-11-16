//
// Created by Mingkai Chen on 2016-11-13.
//

#include "gtest/gtest.h"
#include "mock_ccoms.h"

// test classes

TEST(COMS, observer) {
	div_subject subj;
	div_observer div_obs1(&subj, 4);
	div_observer div_obs2(&subj, 3);
	mod_observer mod_obs3(&subj, 3);
	subj.set_val(14);
	ASSERT_EQ(14/4, div_obs1.get_out());
	ASSERT_EQ(14/3, div_obs2.get_out());
	ASSERT_EQ(14%3, mod_obs3.get_out());
}
