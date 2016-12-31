//
//  tensorshape.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "utils/utils.hpp"

#pragma once
#ifndef tensorshape_hpp
#define tensorshape_hpp

namespace nnet
{

// zero dimension denotes unknown
struct dimension
{
	size_t value_;

	dimension (size_t value) : value_(value) {}
	explicit operator size_t (void) const {
		return value_;
	}

	dimension merge_with (const dimension& other) const;
	bool is_compatible_with (const dimension& other) const;
	void assert_is_compatible_with (const dimension& other) const;
};

class tensorshape
{
	private:
		std::vector<dimension> dimensions_;

	public:
		tensorshape (void) {}
		tensorshape (const std::vector<size_t>& dims);
		tensorshape (const std::vector<dimension>& dims);
		tensorshape& operator = (const std::vector<size_t>& dims);

		// create the most defined tensor
		tensorshape merge_with (const tensorshape& other) const;
		// create a new tensor that is the concatenation of
		// other to this tensor
		tensorshape concatenate (const tensorshape& other) const;
		// create new tensors with the same dimensional value but
		// corresponds to the desired rank
		tensorshape with_rank (size_t rank) const;
		tensorshape with_rank_at_least (size_t rank) const;
		tensorshape with_rank_at_most (size_t rank) const;

		// get the number of elems that can fit into tensorshape, 0 denotes unknown
		size_t n_elems (void) const;
		size_t n_dims (void) const;
		std::vector<dimension> dims (void) const; // deep copy
		std::vector<size_t> as_list (void) const;

		// returns a new shape with leading and padding ones removed
		tensorshape trim (void) const;

		// tensorshape_proto* as_proto (void) const; // serialize

		bool is_compatible_with (const tensorshape& other) const;
		// partially defined if complete
		// (all fully defined are at least partially defined)
		bool is_part_defined (void) const;
		// fully defined if complete with no unknowns
		bool is_fully_defined (void) const;

		// assertions
		void assert_has_rank (size_t rank) const;
		void assert_same_rank (const tensorshape& other) const;
		void assert_is_fully_defined (void) const;

		void undefine (void) { dimensions_.clear(); }
};

void print_shape (tensorshape ts);

}

#endif /* tensorshape_hpp */
