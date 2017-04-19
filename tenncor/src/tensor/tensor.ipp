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
tensor<T>* tensor<T>::clone (void) const { return static_cast<tensor<T>*>(clone_impl()); }

template <typename T>
tensor<T>::tensor (tensor<T>&& other)
{
	move(std::move(other));
}

template <typename T>
tensor<T>& tensor<T>::operator = (const tensor<T>& other)
{
	if (this != &other)
	{
		copy(other);
	}
	return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator = (tensor<T>&& other)
{
	if (this != &other)
	{
		move(std::move(other));
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
optional<tensorshape> tensor<T>::guess_shape (std::vector<T> data) const
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
optional<tensorshape> tensor<T>::loosely_guess_shape(std::vector<T> data) const
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
	size_t rank = alloc_shape_.rank();
	size_t ncoord = coord.size();
	size_t n = std::min(rank, ncoord);
	std::vector<size_t> dims = alloc_shape_.as_list();
	size_t accum = 1;
	size_t raw_idx = 0;
	for (size_t i = 0; i < n; i++)
	{
		raw_idx += coord[i] * accum;
		accum *= dims[i];
	}
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
			raw_copy(temp, shape, raw_data_, alloc_shape_);
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
		raw_copy(temp, shape, other.raw_data_, olds);

		if (is_alloc())
		{
			alloc_->dealloc(raw_data_, alloc_shape_.n_elems());
		}
		raw_data_ = temp;
		alloc_shape_ = shape;
	}
	return success;
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

//template <typename T>
// bool from_proto (const tensorproto& other);

//template <typename T>
// bool from_proto (const tensorproto& other);

template <typename T>
tensor<T>::tensor (const tensor<T>& other)
{
	copy(other);
}

template <typename T>
itensor<T>* tensor<T>::clone_impl (void) const
{
	return new tensor<T>(*this);
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
void tensor<T>::copy (const tensor<T>& other)
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
		std::memcpy(raw_data_, other.raw_data_, sizeof(T) * ns);
	}
}

template <typename T>
void tensor<T>::move (tensor<T>&& other)
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
