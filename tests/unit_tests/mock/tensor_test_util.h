//
// Created by Mingkai Chen on 2016-08-29.
//

#include "gtest/gtest.h"
#include "tensor/tensorshape.hpp"

// tensors equal according to specs
static bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	const nnet::tensorshape& ts2)
{
	if (false == ts1.is_compatible_with(ts2))
	{
		return false;
	}
	return (ts1.is_fully_defined() == ts2.is_fully_defined()) ||
		(ts1.is_part_defined() && ts2.is_part_defined());
}