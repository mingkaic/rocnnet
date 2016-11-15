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
#include "memory/iallocator.hpp"
#include "memory/ram_alloc.hpp"

#pragma once
#ifndef tensor_hpp
#define tensor_hpp

namespace nnet
{

template <typename T>
class assign;
template <typename T>
class gradient;
template <typename T>
class initializer;
template <typename T>
class ivariable;

template <typename T>
std::vector<T> expose (ivariable<T>* var);

static alloc_attrib default_attr; // TODO consider removing to make thread safe

template <typename T>
class tensor
{
	private:
		// meta data (not as template)
		tensorshape allowed_shape_;
		tensorshape alloc_shape_;
		iallocator* alloc_ = nullptr;
		T* raw_data_ = nullptr;

		tensor (const tensor<T>& other);

	protected:
		void copy (const tensor<T>& other);

		virtual tensor<T>* clone_impl (void) { return new tensor<T>(*this); }

		// protected accessor... this isn't overengineering! I swear.
		virtual T* get_raw (void) { return raw_data_; }

		friend class assign<T>;
		friend class gradient<T>;
		friend class initializer<T>;
		friend std::vector<T> expose (ivariable<T>* var);

	public:
		// creates a rank 0 tensor
		tensor (void);
		tensor (tensorshape shape);
		// allocate raw_data on construction
		tensor (tensorshape shape, iallocator& alloc);
		tensor (tensorshape shape, iallocator* alloc);
		tensor (tensorshape shape, iallocator& alloc, const alloc_attrib& attrib);
		tensor (tensorshape shape, iallocator* alloc, const alloc_attrib& attrib);
		// makes a scalar and auto allocates to memory
		tensor (T scalar);
		// rule of three
		virtual ~tensor (void);
		tensor<T>* clone (void) { return clone_impl(); }
		tensor<T>& operator = (const tensor<T>& other);

		// allocate
		// reallocation clear raw data
		void allocate (iallocator* allocer);
		void allocate (iallocator* allocer, const alloc_attrib& attrib);
		void allocate (iallocator* allocer, const tensorshape shape);
		void allocate (iallocator* allocer, const tensorshape shape, const alloc_attrib& attrib);

		// shape info getters
		// get tensor shape
		tensorshape get_shape (void) const;
		// get vector of each dimension
		std::vector<size_t> dims (void) const;
		// get the tensor rank, dims().size()
		size_t n_dims (void) const;
		// get the amount of T elements allocated, 0 if uninitialized
		size_t n_elems (void) const;
		bool is_compatible_with (std::vector<T> data) const;
		bool is_compatible_with (const tensor<T>& other) const;
		// checks if input tensor has a compatible allowed tensorshape
		bool is_same_size (const tensor<T>& other) const;
		// guess shape from the data based on the current shape (allowed if unallocated, allocated otherwise)
		tensorshape guess_shape (std::vector<T> data) const;
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
		// set shape if not allocated
		void set_shape (tensorshape shape);

		// TODO: unimplemented
		bool copy_from (const tensor& other, const tensorshape shape);
		// slice along the first dimension
		tensor<T> slice (size_t dim_start, size_t limit);

		// TODO: read protocol buffers
		// bool shares_buffer_with (const tensor& other) const;
		// size_t buffer_hash (void) const;
		// bool from_proto (const tensorproto& other);
		// bool from_proto (iallocator* a, const tensorproto& other);
};

}

#include "../../src/tensor/tensor.ipp"

#endif /* tensor_hpp */
