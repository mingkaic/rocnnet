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

#include "tensor/itensor.hpp"
#include "tensor/tensorshape.hpp"
#include "memory/default_alloc.hpp"
#include "proto/tenncor.pb.h"

#pragma once
#ifndef TENNCOR_TENSOR_HPP
#define TENNCOR_TENSOR_HPP

#include <stdexcept>
#include <string>
#include <type_traits>
#include <cstring>

namespace nnet
{

//! extremely common function fitting source data represented by inshape to
//! destination data represented by outshape
//! data not covered in source are padded with zero
template <typename T>
void fit_toshape (T* dest, const tensorshape& outshape, const T* src, const tensorshape& inshape);

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

	// >>>> COPY && MOVE <<<<
	//! clone function
	tensor<T>* clone (bool shapeonly = false) const;
	
	//! move function
	tensor<T>* move (void);

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
	void serialize (tenncor::tensor_proto* proto) const;

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
	bool from_proto (const tenncor::tensor_proto& other);

	//! read data and shape from other, reassign allocator
	bool from_proto (const tenncor::tensor_proto& other, size_t alloc_id);

	// slice along the first dimension
	tensor<T> slice (size_t dim_start, size_t limit);

	// bool shares_buffer_with (const tensor& other) const;
	// size_t buffer_hash (void) const;

protected:
	// >>>> COPY, CLONE, && MOVE <<<<
	//! copy constructor
	tensor (const tensor<T>& other, bool shapeonly);

	//! move constructor
	tensor (tensor<T>&& other);

	//! clone implementation
	virtual itensor<T>* clone_impl (bool shapeonly) const;
	
	//! move implementation
	virtual itensor<T>* move_impl (void);

	// >>>> PROTECTED MEMBERS <<<<
	T* raw_data_ = nullptr; //! raw data is available to tensor manipulators

	tensorshape alloc_shape_; //! allocated shape (must be defined)

	friend class itensor_handler<T>;

private:
	//! copy utility helper
	void copy_helper (const tensor<T>& other, bool shapeonly);

	//! move utility helper
	void move_helper (tensor<T>&& other);

	// >>>> PRIVATE MEMBERS <<<<
	//! allocator
	iallocator* alloc_;

	//! not necessarily defined shape
	tensorshape allowed_shape_;
};

}

#include "../../src/tensor/tensor.ipp"

#endif /* TENNCOR_TENSOR_HPP */
