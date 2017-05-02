//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cstring>
#include <functional> // for std::bad_function_call();

#include "memory/alloc_builder.hpp"

#ifdef TENNCOR_TENSOR_HPP

namespace nnet
{

template <typename T>
void fit_toshape (T* dest, const tensorshape& outshape, const T* src, const tensorshape& inshape)
{
	std::vector<size_t> outlist = outshape.as_list();
	std::vector<size_t> inlist = inshape.as_list();
	size_t total = outshape.n_elems();

	memset(dest, 0, sizeof(T) * total);
	std::vector<size_t> fittinglist;
	for (size_t i = 0, n = std::min(total, inlist.size()); i < n; i++)
	{
		fittinglist.push_back(std::min(outlist[i], inlist[i]));
	}
	size_t basewidth = fittinglist[0];
	size_t destidx = 0;
	size_t srcidx = 0;
	size_t n = inshape.n_elems();
	while (srcidx < n)
	{
		// check source index to ensure it is within inlist bounds
		std::vector<size_t> srccoord = inshape.coordinate_from_idx(srcidx);
		bool srcinbound = true;
		size_t extradest = 0;
		size_t src_jump = 1;
		size_t dest_jump = 1;
		for (size_t i = 1, m = fittinglist.size(); srcinbound && i < m; i++)
		{
			srcinbound = srccoord[i] < fittinglist[i];
			if (0 == extradest && srccoord[i] == fittinglist[i]-1 && outlist[i] > inlist[i])
			{
				extradest = dest_jump * outlist[0];
			}
			if (false == srcinbound)
			{
				src_jump *= (inlist[i] - srccoord[i]);
			}
			else
			{
				src_jump *= inlist[i];
			}
			dest_jump *= outlist[i-1];
		}
		if (false == srcinbound)
		{
			srcidx += (src_jump * inlist[0]);
		}
		else
		{
			memcpy(dest + destidx, src + srcidx, sizeof(T) * basewidth);
			srcidx += inlist[0];
			destidx += outlist[0];
		}
		destidx += extradest;
	}
}

template <typename T>
tensor<T>::tensor (void) :
	tensor<T>(std::vector<size_t>{})
{
	set_allocator(default_alloc::alloc_id);
}

template <typename T>
tensor<T>::tensor (T scalar, size_t alloc_id) :
	tensor<T>(std::vector<size_t>{1}, alloc_id)
{
	raw_data_[0] = scalar;
}

template <typename T>
tensor<T>::tensor (tensorshape shape, size_t alloc_id) :
	allowed_shape_(shape)
{
	set_allocator(alloc_id);
	if (allowed_shape_.is_fully_defined())
	{
		alloc_shape_ = shape;
		this->raw_data_ = alloc_->template allocate<T>(
			this->alloc_shape_.n_elems());
	}
}

template <typename T>
tensor<T>::~tensor (void)
{
	alloc_->dealloc(raw_data_, this->alloc_shape_.n_elems());
}

template <typename T>
tensor<T>* tensor<T>::clone (bool shapeonly) const
{
	return static_cast<tensor<T>*>(clone_impl(shapeonly));
}

template <typename T>
tensor<T>* tensor<T>::move (void)
{
	return static_cast<tensor<T>*>(move_impl());
}

template <typename T>
tensor<T>& tensor<T>::operator = (const tensor<T>& other)
{
	if (this != &other)
	{
		copy_helper(other, false);
	}
	return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator = (tensor<T>&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other));
	}
	return *this;
}

template <typename T>
tensorshape tensor<T>::get_shape (void) const
{
	if (is_alloc())
	{
		return alloc_shape_;
	}
	return allowed_shape_;
}

template <typename T>
size_t tensor<T>::n_elems (void) const
{
	if (nullptr == raw_data_)
	{
		return 0;
	}
	return this->alloc_shape_.n_elems();
}

template <typename T>
size_t tensor<T>::rank (void) const
{
	return get_shape().rank();
}

template <typename T>
std::vector<size_t> tensor<T>::dims (void) const
{
	return get_shape().as_list();
}

template <typename T>
bool tensor<T>::is_same_size (const tensor<T>& other) const
{
	if (is_alloc() && other.is_alloc())
	{
		tensorshape simp_shape = alloc_shape_.trim();
		tensorshape other_simp = other.alloc_shape_.trim();
		return simp_shape.is_compatible_with(other_simp);
	}

	return allowed_shape_.is_compatible_with(other.allowed_shape_);
}

template <typename T>
bool tensor<T>::is_compatible_with (const tensor<T>& other) const
{
	return get_shape().is_compatible_with(other.get_shape());
}

template <typename T>
bool tensor<T>::is_compatible_with (std::vector<T> data) const
{
	size_t ndata = data.size();
	const tensorshape& my_shape = is_alloc() ? alloc_shape_ : allowed_shape_;

	bool compatible = true;
	// perfect fit
	if (my_shape.is_fully_defined())
	{
		compatible = ndata == my_shape.n_elems();
	}
	else
	{
		size_t known = my_shape.n_known();
		if (0 < known)
		{
			compatible = 0 == ndata % known;
		}
	}

	return compatible;
}

template <typename T>
bool tensor<T>::is_loosely_compatible_with (std::vector<T> data) const
{
	size_t ndata = data.size();
	const tensorshape& my_shape = is_alloc() ? alloc_shape_ : allowed_shape_;

	bool compatible = true;
	if (my_shape.is_fully_defined())
	{
		compatible = ndata <= my_shape.n_elems();
	}
	// partially defined shapes are always compatible,
	// since unknown dimension can expand infinitely to fit data
	return compatible;
}

template <typename T>
optional<tensorshape> tensor<T>::guess_shape (const std::vector<T>& data) const
{
	size_t ndata = data.size();
	optional<tensorshape> bestshape;
	// if allowed is fully defined
	if (allowed_shape_.is_fully_defined())
	{
		if (allowed_shape_.n_elems() == ndata)
		{
			bestshape = allowed_shape_;
		}
		return bestshape;
	}
	// if allowed is partially defined
	else if (allowed_shape_.is_part_defined())
	{
		std::vector<size_t> my_shape = allowed_shape_.as_list();
		size_t rank = my_shape.size();
		size_t first_undef = my_shape.size();
		size_t known = 1;
		for (size_t i = 0; i < rank; i++)
		{
			if (0 == my_shape[i])
			{
				if (first_undef > i)
				{
					first_undef = i;
				}
				my_shape[i] = 1;
			}
			else
			{
				known *= my_shape[i];
			}
		}
		assert(known > 0);
		if (0 == ndata % known)
		{
			my_shape[first_undef] = ndata / known;
			bestshape = tensorshape(my_shape);
		}
	}
	// if allowed is undefined
	else
	{
		bestshape = tensorshape({ndata});
	}
	return bestshape;
}

template <typename T>
optional<tensorshape> tensor<T>::loosely_guess_shape(const std::vector<T>& data) const
{
	size_t ndata = data.size();
	if (allowed_shape_.is_fully_defined())
	{
		optional<tensorshape> bestshape;
		if (allowed_shape_.n_elems() >= ndata)
		{
			bestshape = allowed_shape_;
		}
		return bestshape;
	}
	std::vector<size_t> my_shape = allowed_shape_.as_list();
	size_t first_undef = my_shape.size();
	size_t known = 1;
	for (size_t i = 0; i < my_shape.size(); i++)
	{
		if (0 == my_shape[i])
		{
			if (first_undef > i)
			{
				first_undef = i;
			}
			my_shape[i] = 1;
		}
		else
		{
			known *= my_shape[i];
		}
	}
	my_shape[first_undef] = ndata / known;
	if (0 != ndata % known)
	{
		// int division above will floor
		// (if we cast to double, we may lose precision)
		my_shape[first_undef]++;
	}
	return tensorshape(my_shape);
}

template <typename T>
bool tensor<T>::is_aligned (void) const { return true; }

template <typename T>
bool tensor<T>::is_alloc (void) const
{
	return alloc_shape_.is_fully_defined() && raw_data_ != nullptr;
}

template <typename T>
size_t tensor<T>::total_bytes (void) const
{
	return n_elems() * sizeof(T);
}

// extension of matrix index representation idx = x+y*col
template <typename T>
T tensor<T>::get (std::vector<size_t> coord) const
{
	size_t raw_idx = alloc_shape_.sequential_idx(coord);
	if (raw_idx >= alloc_shape_.n_elems())
	{
		throw std::out_of_range(nnutils::formatter() <<
		"out of bound coordinate: " << coord);
	}
	return raw_data_[raw_idx];
}

template <typename T>
std::vector<T> tensor<T>::expose (void) const
{
	assert(is_alloc());
	return std::vector<T>(raw_data_, raw_data_ + n_elems());
}

template <typename T>
void tensor<T>::serialize (tenncor::tensor_proto* proto) const
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

template <typename T>
void tensor<T>::set_allocator (size_t alloc_id)
{
	if (iallocator* alloc =
		alloc_builder::get_instance().get(alloc_id))
	{
		alloc_ = alloc;
	}
	else
	{
		throw std::exception(); // todo: better exception
	}
}

template <typename T>
void tensor<T>::set_shape (tensorshape shape)
{
	// allowed shape update
	if (false == allowed_shape_.is_compatible_with(shape) ||
		false == allowed_shape_.is_part_defined()) // always upgrade from undefined
	{
		allowed_shape_ = shape;
	}

	// if shape is compatible with alloc then we don't need to change raw data
	// otherwise we need to modify raw data to match new shape
	if (is_alloc() && false == shape.is_compatible_with(alloc_shape_))
	{
		// if shape isn't defined, we need to make it defined
		// by merging with existing allocated shape
		if (false == shape.is_fully_defined())
		{
			// make alloc_shape compatible with shape
			shape = shape.with_rank(alloc_shape_.rank());
			shape = shape.merge_with(alloc_shape_);
			shape = shape.with_rank(allowed_shape_.rank());
		}
		// shape now represent the desired alloc_shape_
		// reshape by allocate
		allocate(shape);
	}
}

template <typename T>
bool tensor<T>::allocate (void)
{
	bool successful = false;
	if (false == is_alloc())
	{
		// alloc_shape_ can be undefined
		if (alloc_shape_.is_fully_defined())
		{
			successful = allocate(alloc_shape_);
		}
		else
		{
			successful = allocate(allowed_shape_);
		}
	}
	return successful;
}

template <typename T>
bool tensor<T>::deallocate (void)
{
	bool success = is_alloc();
	if (success)
	{
		alloc_->dealloc(raw_data_, alloc_shape_.n_elems());
		raw_data_ = nullptr;
		alloc_shape_.undefine();
	}
	return success;
}

template <typename T>
bool tensor<T>::allocate (const tensorshape shape)
{
	bool success = false;
	if (is_alloc() && shape.is_compatible_with(alloc_shape_))
	{
		return success;
	}
	if (shape.is_compatible_with(allowed_shape_) &&
		shape.is_fully_defined())
	{
		success = true;
		// dealloc before reallocation
		if (is_alloc())
		{
			T* temp = alloc_->template allocate<T>(shape.n_elems());
			// move raw_data to temp matching new shape
			// we want to only copy over the minimum data to lower cost
			fit_toshape(temp, shape, raw_data_, alloc_shape_);
//			raw_copy(temp, shape, raw_data_, alloc_shape_);
			alloc_->dealloc(raw_data_, alloc_shape_.n_elems());
			raw_data_ = temp;
		}
		else
		{
			raw_data_ = alloc_->template allocate<T>(shape.n_elems());
		}
		alloc_shape_ = shape;
	}
	return success;
}

template <typename T>
bool tensor<T>::copy_from (const tensor<T>& other, const tensorshape shape)
{
	bool success = false;
	if (other.is_alloc() &&
		shape.is_fully_defined())
	{
		// allowed shape update
		if (!allowed_shape_.is_compatible_with(shape))
		{
			allowed_shape_ = shape;
		}

		success = true;
		tensorshape olds = other.get_shape();
		T* temp = alloc_->template allocate<T>(shape.n_elems());
		fit_toshape(temp, shape, other.raw_data_, olds);
//		raw_copy(temp, shape, other.raw_data_, olds);

		if (is_alloc())
		{
			alloc_->dealloc(raw_data_, alloc_shape_.n_elems());
		}
		raw_data_ = temp;
		alloc_shape_ = shape;
	}
	return success;
}

template <typename T>
bool tensor<T>::from_proto (const tenncor::tensor_proto& other)
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

template <typename T>
bool tensor<T>::from_proto (const tenncor::tensor_proto& other, size_t alloc_id)
{
	set_allocator(alloc_id);
	return from_proto(other);
}

// slice along the first dimension
template <typename T>
tensor<T> tensor<T>::slice (size_t /*dim_start*/, size_t /*limit*/)
{
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return tensor<T>();
}

//template <typename T>
// bool shares_buffer_with (const tensor<T>& other) const;

//template <typename T>
// size_t tensor<T>::buffer_hash (void) const {
//	 return 0;
// }

template <typename T>
tensor<T>::tensor (const tensor<T>& other, bool shapeonly)
{
	copy_helper(other, shapeonly);
}

template <typename T>
tensor<T>::tensor (tensor<T>&& other)
{
	move_helper(std::move(other));
}

template <typename T>
itensor<T>* tensor<T>::clone_impl (bool shapeonly) const
{
	return new tensor<T>(*this, shapeonly);
}

template <typename T>	
itensor<T>* tensor<T>::move_impl (void)
{
	return new tensor<T>(std::move(*this));
}

template <typename T>
void tensor<T>::raw_copy (T* out, const tensorshape& outs,
   const T* in, const tensorshape& ins) const
{
	size_t resnelem = outs.n_elems();
	size_t oldnelem = ins.n_elems();
	size_t resrow = outs.as_list()[0];
	size_t oldrow = ins.as_list()[0];
	size_t resn = resnelem / resrow;
	size_t oldn = oldnelem / oldrow;
	size_t n = std::min(resn, oldn);
	size_t nrow = std::min(resrow, oldrow);
	// iterate from dimension 2 to dimension n
	// copying over dimension 1 from in to out
	// resn is the product_i=1:n-1(out[i])
	for (size_t i = 0; i < n; i++)
	{
		T* dest = out + i * resrow;
		const T* orig = in + i * oldrow;
		std::memcpy(dest, orig, sizeof(T) * nrow);
		// expand by padding with 0s
		if (resrow > nrow)
		{
			std::memset(dest + nrow, 0, sizeof(T) * (resrow - nrow));
		}
	}
	// if resn is not n, then there are left over memory,
	// fill the leftovers with 0
	if (resn > n)
	{
		std::memset(out + n * resrow, 0, sizeof(T) * resrow * (resn - n));
	}
}

template <typename T>
void tensor<T>::copy_helper (const tensor<T>& other, bool shapeonly)
{
	if (raw_data_)
	{
		alloc_->dealloc(raw_data_, alloc_shape_.n_elems());
		raw_data_ = nullptr;
	}
	alloc_ = other.alloc_;
	alloc_shape_ = other.alloc_shape_;
	allowed_shape_ = other.allowed_shape_;
	if (other.is_alloc())
	{
		size_t ns = alloc_shape_.n_elems();
		raw_data_ = alloc_->template allocate<T>(ns);
		if (false == shapeonly)
			std::memcpy(raw_data_, other.raw_data_, sizeof(T) * ns);
	}
}

template <typename T>
void tensor<T>::move_helper (tensor<T>&& other)
{
	if (raw_data_)
	{
		alloc_->dealloc(raw_data_, alloc_shape_.n_elems());
	}
	// transfer ownership to here.
	raw_data_ = std::move(other.raw_data_);
	// other loses ownership
	other.raw_data_ = nullptr;
	alloc_ = std::move(other.alloc_);
	alloc_shape_ = std::move(other.alloc_shape_);
	allowed_shape_ = std::move(other.allowed_shape_);
}

}

#endif
