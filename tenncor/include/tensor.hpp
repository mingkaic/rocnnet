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
class ievoker;
template <typename T>
class initializer;

static alloc_attrib default_attr; // TODO consider removing to make thread safe

template <typename T>
class tensor {
	private:
		// meta data (not as template)
		tensor_shape allowed_shape_;
		tensor_shape alloc_shape_;
		std::shared_ptr<iallocator> alloc_ = nullptr;

		void copy (const tensor<T>& other);

	protected:
		T* raw_data_ = nullptr; // TODO make unique at some point

		friend class ievoker<T>;
		friend class initializer<T>;

	public:
		std::string name; // identifier

		// creates a rank 0 tensor
		tensor (void);
		tensor (const tensor_shape shape);
		// allocate raw_data on construction
		tensor (std::shared_ptr<iallocator> alloc, const tensor_shape shape);
		tensor (std::shared_ptr<iallocator> alloc, const tensor_shape shape, alloc_attrib const & attrib);
		// makes a scalar
		tensor (T scalar);
		// rule of three
		tensor (const tensor<T>& other);
		virtual ~tensor (void);
		tensor<T>& operator = (const tensor<T>& other);

		// allocate
		// reallocation clear raw data
		void allocate (std::shared_ptr<iallocator> allocer);
		void allocate (std::shared_ptr<iallocator> allocer, const alloc_attrib& attrib);
		void allocate (std::shared_ptr<iallocator> allocer, const tensor_shape shape);
		void allocate (std::shared_ptr<iallocator> allocer,
						const tensor_shape shape,
						const alloc_attrib& attrib);

		// shape info getters
		// get tensor shape
		tensor_shape get_shape (void) const;
		// get vector of each dimension
		std::vector<size_t> dims (void) const;
		// get the tensor rank, dims().size()
		size_t n_dims (void) const;
		// get the amount of T elements allocated, 0 if uninitialized
		size_t n_elems (void) const;
		bool is_compatible_with (std::vector<T> data) const;
		bool is_compatible_with (const tensor<T>& other) const;
		// checks if input tensor has a compatible allowed tensor_shape
		bool is_same_size (const tensor<T>& other) const;
		// guess shape from the data based on the current shape (allowed if unallocated, allocated otherwise)
		tensor_shape guess_shape (std::vector<T> data) const;
		// checks if tensorshape is aligned
		// (e.g.: same number of column for each row)
		bool is_aligned (void) const;

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
		bool copy_from (const tensor& other, const tensor_shape shape);
		// slice along the first dimension
		tensor slice (size_t dim_start, size_t limit);

		// bool shares_buffer_with (const tensor& other) const;
		// size_t buffer_hash (void) const;
		// bool from_proto (const tensorproto& other);
		// bool from_proto (iallocator* a, const tensorproto& other);
};

template <typename T>
using TENSOR_PTR = std::shared_ptr<tensor<T> >;

}

#include "../src/tensor.ipp"

#endif /* tensor_hpp */
