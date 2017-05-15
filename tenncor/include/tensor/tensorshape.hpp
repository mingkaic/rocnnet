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

	// (UNTESTED)
	// >>>> COORDINATES <<<<
	//! obtains the index of data should they be flatten to a vector given dimensions_
	size_t sequential_idx (std::vector<size_t> coord) const;

	//! obtain the coordinates of the data in vector given a sequential index given dimensions_
	std::vector<size_t> coordinate_from_idx (size_t idx) const;

	//! obtains the set of indices corresponding to elements in memory (grouped elements)
	//! shapeidx is index obtained from sequential_idx based on dimensions_
	std::vector<size_t> memory_indices (size_t shapeidx) const;

	//! obtains the dimensional values not accounting for grouping
	std::vector<size_t> shape_dimensions (void) const;

	//! determine whether shape is grouped at some diemnsion
	bool is_grouped (void) const;

	//! group shape at specified dimension
	void group_dim (size_t dim);

private:
	//! zero values denotes unknown/undefined value
	//! emtpy dimension_ denotes undefined shape
	std::vector<size_t> dimensions_;

	//! determines which dimension the shape is grouped and by how much
	//! having a dim_group indicates that shape index and memory index do not correspond 1 to 1
	//! first value denotes the dimensional index where grouping is occurring
	//! second value denotes the number of real elements per grouping
	//! (where the real dimension is dimensions_[first] *= second)
	optional<std::pair<size_t,size_t> > dim_group_;
};

//! print a shape's dimension values
void print_shape (tensorshape ts, std::ostream& os = std::cout);

}

#endif /* TENNCOR_TENSORSHAPE_HPP */
