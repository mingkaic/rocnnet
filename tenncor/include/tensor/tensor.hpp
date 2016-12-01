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
#include <type_traits>
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
template <typename T, typename A>
class tensor_op;

static alloc_attrib default_attr; // TODO consider removing to make thread safe

template <typename T, typename A=ram_alloc>
class tensor
{
	static_assert(std::is_base_of<iallocator, A>(), "allocator must inherit from iallocator");
	private:
		// meta data (not as template)
		tensorshape allowed_shape_;
		tensorshape alloc_shape_;
		const A alloc_;
		T* raw_data_ = nullptr;

	protected:
		void copy (const tensor<T,A>& other);
		tensor (const tensor<T,A>& other);

		virtual tensor<T,A>* clone_impl (void) { return new tensor<T,A>(*this); }

		// protected accessor... this isn't overengineering! I swear.
		virtual T* get_raw (void) { return raw_data_; }

		// forcefully deallocates raw_data (reserved for children only... hopefully)
		void change_shape (tensorshape res_shape)
		{
			// deallocate if raw_data_ is not null (allocator checks for raw nulls)
			alloc_.dealloc(raw_data_, alloc_shape_.n_elems());
			raw_data_ = nullptr;
			// reshape allowed_shape_
			set_shape(res_shape);
			alloc_shape_.undefine(); // make undefine since we're no longer allocated
		}

		friend class assign<T>;
		friend class gradient<T>;
		friend class initializer<T>;
		friend class tensor_op<T,A>;

		template <typename U>
		friend std::vector<U> expose (ivariable<U>* var);
		
		template <typename U>
		friend std::vector<U> expose (tensor<U>* ten);

	public:
		// creates a rank 0 tensor
		tensor (void);
		// allocate on construction if shape is fully defined
		tensor (tensorshape shape);
		tensor (tensorshape shape, const alloc_attrib& attrib);
		// makes a scalar and auto allocates to memory
		tensor (T scalar);
		// rule of three
		virtual ~tensor (void);
		tensor<T,A>* clone (void) { return clone_impl(); }
		tensor<T,A>& operator = (const tensor<T,A>& other);
		// act on tensor
		virtual const tensor<T,A>& operator () (std::vector<tensor<T,A>*> args) { return *this; }

		// allocate
		// reallocation clear raw data
		void allocate (void);
		void allocate (const alloc_attrib& attrib);
		void allocate (const tensorshape shape);
		void allocate (const tensorshape shape, const alloc_attrib& attrib);

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
		bool is_compatible_with (const tensor<T,A>& other) const;
		// checks if input tensor has a compatible allowed tensorshape
		bool is_same_size (const tensor<T,A>& other) const;
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
		virtual T get (std::vector<size_t> indices);

		// setter
		// set shape if not allocated
		void set_shape (tensorshape shape);

		// TODO: unimplemented
		bool copy_from (const tensor& other, const tensorshape shape);
		// slice along the first dimension
		tensor<T,A> slice (size_t dim_start, size_t limit);

		// TODO: read protocol buffers
		// bool shares_buffer_with (const tensor& other) const;
		// size_t buffer_hash (void) const;
		// bool from_proto (const tensorproto& other);
		// bool from_proto (iallocator* a, const tensorproto& other);
		virtual void raw_update (void) {}
};

template <typename T>
std::vector<T> expose (tensor<T>* ten)
{
	assert(nullptr != ten);
	T* raw = ten->get_raw();
	assert(ten->is_alloc()); // assert after get_raw to allow tensor a chance
	return std::vector<T>(raw, raw + ten->n_elems());
}

}

#include "../../src/tensor/tensor.ipp"

#endif /* tensor_hpp */
