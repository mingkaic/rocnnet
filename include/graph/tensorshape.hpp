//
//  tensorshape.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cassert>
#include <stdexcept>
#include <vector>

#pragma once
#ifndef tensorshape_hpp
#define tensorshape_hpp

namespace nnet {

// zero dimension denotes unknown
struct dimension {
	size_t value;

	dimension (size_t value) : value(value) {}
	explicit operator size_t() { return value; }

	dimension merge_with (dimension const & other) const;
	bool is_compatible_with (dimension const & other) const;
	void assert_is_compatible_with (dimension const & other) const;
};

class tensor_shape {
	private:
		std::vector<dimension> dimensions;

	public:
		tensor_shape (void) {}
		tensor_shape (std::vector<size_t> const & dims);
		tensor_shape (std::vector<dimension> const & dims);
		tensor_shape& operator = (std::vector<size_t> const & dims);
		tensor_shape merge_with (tensor_shape const & other);
		tensor_shape concatenate (tensor_shape const & other);
		tensor_shape with_rank (size_t rank);
		tensor_shape with_rank_at_least (size_t rank);
		tensor_shape with_rank_at_most (size_t rank);

		// get the number of elems that can fit into tensorshape, 0 denotes unknown
		size_t n_elems (void) const;
		size_t n_dims (void) const;
		std::vector<dimension> dims (void) const; // deep copy
		std::vector<size_t> as_list (void) const;

		// tensor_shape_proto* as_proto (void) const; // serialize

		bool is_compatible_with (tensor_shape const & other) const;
		// partially defined if complete
		// (all fully defined are at least partially defined)
		bool is_part_defined (void) const;
		// fully defined if complete with no unknowns
		bool is_fully_defined (void) const;

		// assertions
		void assert_has_rank (size_t rank) const;
		void assert_same_rank (tensor_shape const & other) const;
		void assert_is_fully_defined (void) const { assert(is_fully_defined()); }
};

}

#endif /* tensorshape_hpp */
