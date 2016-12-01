//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cstring>
#include <functional> // for std::bad_function_call();

#ifdef tensor_hpp

namespace nnet
{

template <typename T, typename A>
void tensor<T,A>::copy (const tensor<T,A>& other)
{
	T* other_raw = other.raw_data_;
	if (1 != other.n_elems()) // not a scalar
	{
		// allocate if other is allocated
		if (nullptr != other_raw)
		{
			assert(this->is_compatible_with(other));
			if (is_alloc())
			{
				alloc_.dealloc(raw_data_, this->alloc_shape_.n_elems());
			}
			size_t nelem = other.n_elems();
			// modify depending on raw_data_ implementation
			raw_data_ = alloc_.template allocate<T>(nelem, alloc_attrib());
			std::memcpy(raw_data_, other_raw, sizeof(T) * nelem);
		}
		// shape info copies over regardless
		allowed_shape_ = other.allowed_shape_;
		alloc_shape_ = other.alloc_shape_;
	}
	else // fill current tensor with scalar value
	{
		assert(other_raw); // other must be allocated
		T scalar = other_raw[0];
		
		if (false == is_alloc()) // this must be allocated
		{
			allocate();
		}
		
		size_t ns = n_elems();
		std::fill(raw_data_, raw_data_+ns, scalar);
	}
}

template <typename T, typename A>
tensor<T,A>::tensor (const tensor<T,A>& other) : alloc_()
{
	this->copy(other);
}

template <typename T, typename A>
tensor<T,A>::tensor (void) : tensor<T,A>(std::vector<size_t>{}) {}

template <typename T, typename A>
tensor<T,A>::tensor (tensorshape shape) :
	allowed_shape_(shape), alloc_()
{
	if (allowed_shape_.is_fully_defined())
	{
		alloc_shape_ = shape;
		this->raw_data_ = alloc_.template allocate<T>(this->alloc_shape_.n_elems(), default_attr);
	}
}

template <typename T, typename A>
tensor<T,A>::tensor (tensorshape shape, const alloc_attrib& attrib) :
	allowed_shape_(shape), alloc_()
{
	if (allowed_shape_.is_fully_defined())
	{
		alloc_shape_ = shape;
		this->raw_data_ = alloc_.template allocate<T>(this->alloc_shape_.n_elems(), attrib);
	}
}

template <typename T, typename A>
tensor<T,A>::tensor (T scalar) :
	tensor<T,A>(std::vector<size_t>{1})
{
	this->raw_data_[0] = scalar;
}

template <typename T, typename A>
tensor<T,A>::~tensor (void)
{
	alloc_.dealloc(raw_data_, this->alloc_shape_.n_elems());
}

template <typename T, typename A>
tensor<T,A>& tensor<T,A>::operator = (tensor<T,A>& other)
{
	if (this != &other)
	{
		other.raw_update();
		this->copy(other);
	}
	return *this;
}

template <typename T, typename A>
void tensor<T,A>::allocate (void)
{
	// alloc_shape_ can be undefined
	if (false == alloc_shape_.is_fully_defined())
	{
		allocate(allowed_shape_, default_attr);
	}
	else
	{
		allocate(alloc_shape_, default_attr);
	}
}

template <typename T, typename A>
void tensor<T,A>::allocate (const alloc_attrib& attrib)
{
	allocate(allowed_shape_, attrib);
}

template <typename T, typename A>
void tensor<T,A>::allocate (const tensorshape shape)
{
	allocate(shape, default_attr);
}

template <typename T, typename A>
void tensor<T,A>::allocate (const tensorshape shape, const alloc_attrib& attrib)
{
	assert(shape.is_compatible_with(allowed_shape_));
	shape.assert_is_fully_defined();

	// dealloc before reallocation
	if (true == is_alloc())
	{
		alloc_.dealloc(raw_data_, this->alloc_shape_.n_elems());
	}

	alloc_shape_ = shape;
	raw_data_ = alloc_.template allocate<T>(this->alloc_shape_.n_elems(), attrib);
}

template <typename T, typename A>
tensorshape tensor<T,A>::get_shape (void) const
{
	if (is_alloc())
	{
		return alloc_shape_;
	}
	return allowed_shape_;
}

template <typename T, typename A>
std::vector<size_t> tensor<T,A>::dims (void) const
{
	if (is_alloc())
	{
		return alloc_shape_.as_list();
	}
	return allowed_shape_.as_list();
}

template <typename T, typename A>
size_t tensor<T,A>::n_dims (void) const { return allowed_shape_.n_dims(); }

template <typename T, typename A>
size_t tensor<T,A>::n_elems (void) const
{
	if (nullptr == raw_data_)
	{
		return 0;
	}
	return this->alloc_shape_.n_elems();
}

// always aligned... until I add unaligned tensorshape
// extend tensorshape which is always aligned
template <typename T, typename A>
bool tensor<T,A>::is_aligned (void) const
{
	return true;
}

template <typename T, typename A>
tensorshape tensor<T,A>::guess_shape (std::vector<T> data) const
{
	if (allowed_shape_.is_fully_defined())
	{
		return allowed_shape_;
	}
	std::vector<size_t> my_shape = allowed_shape_.as_list();
	size_t first_undef = my_shape.size();
	size_t fixed = 1;
	for (size_t i = 0; i < my_shape.size(); i++)
	{
		if (0 == my_shape[i])
		{
			if (first_undef > i) first_undef = i;
			my_shape[i] = 1;
		}
		else
		{
			fixed *= my_shape[i];
		}
	}
	my_shape[first_undef] = data.size() / fixed;
	return my_shape;
}

template <typename T, typename A>
bool tensor<T,A>::is_compatible_with (std::vector<T> data) const
{
	std::vector<size_t> my_shape = allowed_shape_.as_list();
	if (is_alloc())
	{
		my_shape = alloc_shape_.as_list();
	}
	size_t fixed = 1;
	for (size_t s : my_shape)
	{
		if (s) fixed *= s;
	}
	// not part defined means fully undefined, undefined is compatible with any data type
	return !allowed_shape_.is_part_defined() || 0 == data.size() % fixed;
}

template <typename T, typename A>
bool tensor<T,A>::is_compatible_with (const tensor<T,A>& other) const
{
	return get_shape().is_compatible_with(other.get_shape());
}

template <typename T, typename A>
bool tensor<T,A>::is_same_size (const tensor<T,A>& other) const
{
	tensorshape simp_shape = alloc_shape_.trim();
	tensorshape other_simp = other.alloc_shape_.trim();

	return (this->is_alloc() && other.is_alloc() &&
			simp_shape.is_compatible_with(other_simp)) ||
		   (this->allowed_shape_.is_compatible_with(other.allowed_shape_));
}

template <typename T, typename A>
bool tensor<T,A>::is_alloc (void) const
{
	return alloc_shape_.is_fully_defined() && raw_data_ != nullptr;
}

template <typename T, typename A>
size_t tensor<T,A>::total_bytes (void) const
{
	return n_elems() * sizeof(T);
}

// extension of matrix index representation idx = x+y*col
template <typename T, typename A>
T tensor<T,A>::get (std::vector<size_t> indices)
{
	T* raw = this->get_raw();
	size_t rank = alloc_shape_.n_dims();
	if (indices.size() > rank)
	{
		throw std::logic_error(
			nnutils::formatter() << "eliciting extraneous dimensions from a tensor of rank " << rank);
	}
	std::vector<size_t> dims = alloc_shape_.as_list();
	size_t accum = 1;
	size_t raw_idx = 0;
	for (size_t i = 0; i < indices.size(); i++)
	{
		raw_idx += indices[i] * accum;
		accum *= dims[i];
	}
	return raw[raw_idx];
}

template <typename T, typename A>
void tensor<T,A>::set_shape (tensorshape shape)
{
	assert(false == is_alloc()); // impacts allocated raw data
	this->allowed_shape_ = shape;
}

// how to handle shape expansion / compression?
template <typename T, typename A>
bool tensor<T,A>::copy_from (const tensor<T,A>& other, const tensorshape shape)
{
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return false;
}

// slice along the first dimension
template <typename T, typename A>
tensor<T,A> tensor<T,A>::slice (size_t dim_start, size_t limit)
{
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return tensor<T,A>();
}

// bool shares_buffer_with (const tensor<T,A>& other) const;

//template <typename T, typename A>
// size_t tensor<T,A>::buffer_hash (void) const {
//	 return 0;
// }

// bool from_proto (const tensorproto& other);

// bool from_proto (const tensorproto& other);

}

#endif
