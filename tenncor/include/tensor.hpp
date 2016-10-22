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
#include "memory/allocator.hpp"

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
class variable;

template <typename T>
class tensor {
	private:
		// meta data (not as template)
		tensor_shape allowed_shape;
		tensor_shape alloc_shape;

		void copy (const tensor<T>& other);

	protected:
		iallocator* alloc = nullptr;
		T* raw_data = nullptr;

		friend class initializer<T>;
		friend class ioperation<T>;
		// TODO make a better, unified method of manipulating tensors from variable (non-operations objects like scalar and variable)
		friend class variable<T>;

	public:
		// name identifier
		std::string name;

		// creates a rank 0 tensor
		tensor (void);
		tensor (const tensor_shape& shape);
		// allocate raw_data on construction
		tensor (iallocator& a, const tensor_shape& shape)
		: tensor(a, shape, default_attr) {}
		tensor (iallocator& a,
			tensor_shape const & shape,
			alloc_attrib const & attrib);

		// rule of three
		tensor (const tensor<T>& other);
		virtual ~tensor (void);
		tensor<T> & operator = (const tensor<T>& other);

		// allocate
		// reallocation clear raw data
		void allocate (iallocator& allocer);
		void allocate (iallocator& allocer, const alloc_attrib& attrib);
		void allocate (iallocator& allocer, const tensor_shape& shape);
		void allocate (iallocator& allocer, const tensor_shape& shape,
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

		tensor_shape guess_shape (std::vector<T> data) const;
		bool is_compatible_with (std::vector<T> data) const;
		bool is_compatible_with (const tensor<T>& other) const;
		// checks if input tensor has a compatible allowed tensor_shape
		bool is_same_size (const tensor<T>& other) const;

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
		bool copy_from (const tensor& other, const tensor_shape& shape);
		// slice along the first dimension
		tensor slice (size_t dim_start, size_t limit);

		// bool shares_buffer_with (const tensor& other) const;
		// size_t buffer_hash (void) const;
		// bool from_proto (const tensorproto& other);
		// bool from_proto (iallocator* a, const tensorproto& other);
};

}

#include "../src/tensor.tpp"

#endif /* tensor_hpp */
