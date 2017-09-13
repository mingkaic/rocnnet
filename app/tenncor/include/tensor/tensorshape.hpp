/*!
 *
 *  tensorshape.hpp
 *  cnnet
 *
 *  Purpose:
 *  tensorshape stores aligned shape information
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "utils/utils.hpp"

#pragma once
#ifndef TENNCOR_TENSORSHAPE_HPP
#define TENNCOR_TENSORSHAPE_HPP

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <experimental/optional>

using namespace std::experimental;

namespace nnet
{

class tensorshape final
{
public:
	//! by default create a rankless shape
	tensorshape (void) {}

	//! create a shape with the desired dimensions
	tensorshape (const std::vector<size_t>& dims);

	//! assign the desired dimensions to this shape
	tensorshape& operator = (const std::vector<size_t>& dims);

	// >>>> ACCESSORS <<<<
	//! get a copy of the shape as a list
	//! accounts for grouping
	std::vector<size_t> as_list (void) const;

	//! get number of elems that can fit in shape if known,
	//! 0 if unknown
	//! accounts for grouping
	size_t n_elems (void) const;

	//! get the minimum number of elements that can fit in shape
	//! return the product of all known dimensions
	//! accounts for grouping
	size_t n_known (void) const;

	//! get shape rank
	size_t rank (void) const;

	//! check if the shape is compatible with other
	//! does not accounts for grouping
	bool is_compatible_with (const tensorshape& other) const;

	//! check if shape is partially defined
	//! (if there are unknowns but rank is not 0)
	bool is_part_defined (void) const;

	//! check if shape is  fully defined
	//! (there are no unknowns)
	bool is_fully_defined (void) const;

	// >>>> ASSERT <<<<
	//! assert if shape has specified rank
	void assert_has_rank (size_t rank) const;

	//! assert if shape has same rank as other shape
	void assert_same_rank (const tensorshape& other) const;

	//! assert if shape is fully defined
	void assert_is_fully_defined (void) const;

	// >>>> MUTATORS <<<<
	//! invalidate shape
	void undefine (void);

	// >>>> SHAPE CREATORS <<<<
	//! create the most defined shape from this and other
	//! prioritizes this over other value
	tensorshape merge_with (const tensorshape& other) const;

	//! create a copy of this shape with leading and
	//! trailing ones removed
	tensorshape trim (void) const;

	//! create a shape that is the concatenation of another shape
	tensorshape concatenate (const tensorshape& other) const;

	//! create a new tensors with same dimension
	//! value and the specified rank
	//! clip or pad with 1's to fit rank
	tensorshape with_rank (size_t rank) const;

	//! create a new tensors with same dimension
	//! value and at least the the specified rank
	tensorshape with_rank_at_least (size_t rank) const;

	//! create a new tensors with same dimension
	//! value and at most the the specified rank
	tensorshape with_rank_at_most (size_t rank) const;

	// todo: test
	// >>>> COORDINATES <<<<
	//! obtain the flat vector index from cartesian coordinates (e.g.: 2-D [x, y] has flat index = y * dimensions_[0] + x)
	size_t flat_idx (std::vector<size_t> coord) const;

	//! obtain cartesian coordinates given a flat vector index
	std::vector<size_t> coordinate_from_idx (size_t idx) const;

	//! iterate with cartesian and flat coordinates of every valid element in shape
	void iterate (std::function<void(std::vector<size_t>, size_t)> coord_call) const;

private:
	//! zero values denotes unknown/undefined value
	//! emtpy dimension_ denotes undefined shape
	std::vector<size_t> dimensions_;
};

//! print a shape's dimension values
void print_shape (tensorshape ts, std::ostream& os = std::cout);

}

#endif /* TENNCOR_TENSORSHAPE_HPP */
