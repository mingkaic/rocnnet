//
//  tensor.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <stdexcept>
#include <string>
#include <vector>

#include "tensorshape.hpp"
#include "allocator.hpp"

#pragma once
#ifndef tensor_hpp
#define tensor_hpp

namespace nnet {

template <typename T>
using BUFFER = std::vector<T>;

static alloc_attrib default_attr;

template <typename T>
class initializer;

template <typename T>
class ioperation;

template <typename T>
class tensor {
	private:
		// meta data (not as template)
		tensor_shape allowed_shape;
		tensor_shape alloc_shape;

		void copy (tensor<T> const & other);

	protected:
		iallocator* alloc = nullptr;
		T* raw_data = nullptr;

		friend class initializer<T>;
		friend class ioperation<T>;

	public:
		// name identifier
		std::string name;

		// creates a rank 0 tensor
		tensor (void);
		tensor (tensor_shape const & shape);
		tensor (iallocator& a, tensor_shape const & shape)
		: tensor(a, shape, default_attr) {}
		tensor (iallocator& a,
			tensor_shape const & shape,
			alloc_attrib const & attrib);

		// rule of three
		tensor (tensor<T> const & other);
		virtual ~tensor (void);
		tensor<T> & operator = (tensor<T> const & other);

		// allocate
		// reallocation clear raw data
		void allocate (iallocator& allocer);
		void allocate (iallocator& allocer, alloc_attrib const & attrib);
		void allocate (iallocator& allocer, tensor_shape const & shape);
		void allocate (iallocator& allocer, tensor_shape const & shape,
			alloc_attrib const & attrib);

		// shape info getters
		// get tensor shape
		tensor_shape get_shape (void) const;
		// get vector of each dimension
		std::vector<size_t> dims (void) const;
		// get the tensor rank, dims().size()
		size_t n_dims (void) const;
		// get the amount of T elements allocated, 0 if uninitialized
		size_t n_elems (void) const;
		// checks if tensorshape is aligned
		// (e.g.: same number of column for each row)
		bool is_aligned (void) const;
		// checks if input tensor has a compatible allowed tensor_shape
		bool is_same_size (tensor<T> const & other) const;

		// memory info getter
		// check if memory is allocated
		bool is_alloc (void) const;
		// get bytes allocated
		size_t total_bytes (void) const;
		// get data at indices
		T get (std::vector<size_t> indices) const;

		// setter
		// set shape
		void set_shape (tensor_shape shape);

		// TODO: unimplemented
		bool copy_from (tensor const & other, tensor_shape const & shape);
		// slice along the first dimension
		tensor slice (size_t dim_start, size_t limit);

		// bool shares_buffer_with (tensor const & other) const;
		// size_t buffer_hash (void) const;
		// bool from_proto (tensorproto const & other);
		// bool from_proto (iallocator* a, tensorproto const & other);
};

}

#include "../../src/graph/tensor.tpp"

#endif /* tensor_hpp */
