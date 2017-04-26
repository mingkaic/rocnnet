/*!
 *
 *  tensor.hpp
 *  cnnet
 *
 *  Purpose:
 *  tensor object manages shape and raw data
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <stdexcept>
#include <string>
#include <type_traits>

#include "tensor/itensor.hpp"
#include "tensor/tensorshape.hpp"
#include "memory/default_alloc.hpp"
#include "proto/tenncor.pb.h"

#pragma once
#ifndef TENNCOR_TENSOR_HPP
#define TENNCOR_TENSOR_HPP

namespace nnet
{

template <typename T>
class itensor_handler;

template <typename T>
class tensor : public itensor<T>
{
public:
	//! create a rank 0 tensor and specific allocator
	tensor (void);

	//! Create a scalar tensor and specific allocator
	tensor (T scalar, size_t alloc_id = default_alloc::alloc_id);

	//! create a tensor of a specified shape and allocator
	//! if the shape is fully defined, then raw data is allocated
	//! otherwise, tensor will wait for a defined shape
	tensor (tensorshape shape, size_t alloc_id = default_alloc::alloc_id);

	//! deallocate tensor
	virtual ~tensor (void);

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function
	tensor<T>* clone (void) const;

	//! move constructor
	tensor (tensor<T>&& other);

	//! copy assignment
	virtual tensor<T>& operator = (const tensor<T>& other);

	//! move assignment
	virtual tensor<T>& operator = (tensor<T>&& other);

	// >>>> ACCESSORS <<<<
	// >>> SHAPE INFORMATION <<<
	//! get tensor shape (allocated if so, allowed shape otherwise)
	tensorshape get_shape (void) const;

	// >> SHAPE UTILITY <<
	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	virtual size_t n_elems (void) const;

	//! get the tensor rank, number of dimensions
	virtual size_t rank (void) const;

	//! get vector dimension values
	std::vector<size_t> dims (void) const;

	// >> SHAPE COMPATIBILITY <<
	//! checks if input tensor has a compatible allowed tensorshape
	//! or if both this and other are allocated and the trimmed shapes are compatible
	bool is_same_size (const tensor<T>& other) const;

	//! check if other tensor's data is compatible with this shape
	bool is_compatible_with (const tensor<T>& other) const;

	//! check if input is compatible with tensor shape
	//! data is compatible if data.size() == (innate or external) shape size
	bool is_compatible_with (std::vector<T> data) const;

	//! data is loosely compatible if data.size() < (innate or external) shape size
	bool is_loosely_compatible_with (std::vector<T> data) const;

	//! return compatible shape with n_elems == data.size()
	//! or undefined if compatibility is impossible
	// implementation detail:
	// this algorithm attempts to cover up the first unknown with data.size() / n_known
	// iff data.size() % n_known == 0
	// todo: attempt to parameterize some lambda function to distribute data.size() / n_known amongst all unknown (same for loosely guess)
	optional<tensorshape> guess_shape (const std::vector<T>& data) const;

	//! return loosely compatible shape with n_elems <= data.size()
	//! or undefined if compatibility is impossible
	optional<tensorshape> loosely_guess_shape (const std::vector<T>& data) const;

	//! checks if tensorshape is aligned
	//! same number of column for each row
	virtual bool is_aligned (void) const;

	// >>> DATA INFORMATION <<<
	//! checks if memory is allocated
	virtual bool is_alloc (void) const;

	//! get bytes allocated
	virtual size_t total_bytes (void) const;

	//! get data at coordinate specified
	//! getting out of bound will throw out_of_range error
	//! coordinate values not specified are implied as 0
	virtual T get (std::vector<size_t> coord) const;

	//! exposing unallocated shape will cause assertion death
	//! otherwise return data array copy
	std::vector<T> expose (void) const;

	//! serialize protobuf tensor
	void serialize (tenncor::tensor_proto* proto) const
	{
		if (false == is_alloc()) return;
		// copy bytes
		size_t nb = total_bytes();
		proto->set_data(raw_data_, nb);

		std::vector<size_t> allow = allowed_shape_.as_list();
		std::vector<size_t> alloc = alloc_shape_.as_list();
		google::protobuf::RepeatedField<uint64_t> allow_field(allow.begin(), allow.end());
		google::protobuf::RepeatedField<uint64_t> alloc_field(alloc.begin(), alloc.end());

		proto->mutable_allow_shape()->Swap(&allow_field);
		proto->mutable_alloc_shape()->Swap(&alloc_field);
	}

	// >>>> MUTATOR <<<<
	//! get allocator from factory and set it as alloc_
	void set_allocator (size_t alloc_id);

	//! set a new allowed shape
	//! chop raw data outside of new shape
	//! worst case runtime: O(min(N, M))
	//! where N is the original shape size
	//! and M is the resulting shape size
	//! result is shape is compatible with allowed shape
	void set_shape (tensorshape shape);

	//! allocate raw data using allowed (innate) shape
	//! return true if successful
	virtual bool allocate (void);

	//! forcefully deallocate raw_data,
	//! invalidates allocated (external) shape
	//! could be useful when we want to preserve allowed shape
	//! since get_shape when allocated gives allocated shape
	virtual bool deallocate (void);

	//! allocate raw data using input shape
	//! if shape is compatible with allowed
	//! else return false
	virtual bool allocate (const tensorshape shape);

	//! copy raw_data from other expanded/compressed to input shape
	//! allowed shape will be adjusted similar to set_shape
	bool copy_from (const tensor& other, const tensorshape shape);

	//! read data and shape from other, take allocator as is
	bool from_proto (const tenncor::tensor_proto& other)
	{
		std::string protostr = other.data();
		const char* protoraw = protostr.c_str();
		// shapes must have same dimensionality... (otherwise, input data is definitely corrupt)
		assert(other.alloc_shape_size() == other.allow_shape_size());
		std::vector<size_t> allow;
		std::vector<size_t> alloc;
		for (size_t i = 0, n = other.allow_shape_size(); i < n; i++)
		{
			allow.push_back(other.allow_shape(i));
			alloc.push_back(other.alloc_shape(i));
		}
		allowed_shape_ = tensorshape(allow);
		tensorshape temp_alloc_shape(alloc);
		// another sanity check, be less stringent, since this may represent some less evident issue
		if (false == temp_alloc_shape.is_compatible_with(allowed_shape_) ||
			false == temp_alloc_shape.is_fully_defined()) return false;

		deallocate();
		alloc_shape_ = temp_alloc_shape;
		assert(allocate());

		// copy data over from protoraw
		std::memcpy(raw_data_, protoraw, protostr.size());

		return true;
	}

	//! read data and shape from other, reassign allocator
	bool from_proto (const tenncor::tensor_proto& other, size_t alloc_id)
	{
		set_allocator(alloc_id);
		return from_proto(other);
	}

	// slice along the first dimension
	tensor<T> slice (size_t dim_start, size_t limit);

	// bool shares_buffer_with (const tensor& other) const;
	// size_t buffer_hash (void) const;

protected:
	// >>>> COPY && CLONE <<<<
	//! copy constructor
	tensor (const tensor<T>& other);

	//! clone implementation
	virtual itensor<T>* clone_impl (void) const;

	// >>>> PROTECTED MEMBERS <<<<
	T* raw_data_ = nullptr; //! raw data is available to tensor manipulators

	tensorshape alloc_shape_; //! allocated shape (must be defined)

	friend class itensor_handler<T>;

private:
	//! tensor reshape utility
	void raw_copy (T* out, const tensorshape& outs,
		const T* in, const tensorshape& ins) const;

	//! copy utility helper
	void copy (const tensor<T>& other);

	//! move utility helper
	void move (tensor<T>&& other);

	// >>>> PRIVATE MEMBERS <<<<
	//! allocator
	iallocator* alloc_;

	//! not necessarily defined shape
	tensorshape allowed_shape_;
};

}

#include "../../src/tensor/tensor.ipp"

#endif /* TENNCOR_TENSOR_HPP */
