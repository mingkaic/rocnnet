//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef TENNCOR_MOCK_TENSOR_H
#define TENNCOR_MOCK_TENSOR_H


#include "util_test.h"

//#include "tensor/tensor.hpp"


template <typename T>
class tensor {
public:
	//! create a rank 0 tensor
	tensor(void) {}

	//! Create a scalar tensor
	tensor(T scalar) {}

	//! create a tensor of a specified shape and allocator
	//! if the shape is fully defined, then raw data is allocated
	//! otherwise, tensor will wait for a defined shape
	tensor(tensorshape shape) {}

	tensor (const tensor<T>& other) {}

	//! deallocate tensor
	virtual ~tensor(void) {}

	// >>>> CLONE, MOVE, && COPY ASSIGNMENT <<<<
	//! clone function
	tensor<T> *clone(void) const { return nullptr; }

	//! move constructor
	tensor(tensor<T> &&other) {}

	//! copy assignment
	virtual tensor<T> &operator=(const tensor<T> &other) { return *this; }

	//! move assignment
	virtual tensor<T> &operator=(tensor<T> &&other) { return *this; }

	// >>>> ACCESSORS <<<<
	// >>> SHAPE INFORMATION <<<
	//! get tensor shape (allocated if so, allowed shape otherwise)
	tensorshape get_shape(void) const { return tensorshape(); }

	// >> SHAPE UTILITY <<
	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	size_t n_elems(void) const { return 0; }

	//! get the tensor rank, number of dimensions
	size_t rank(void) const { return 0; }

	//! get vector dimension values
	std::vector<size_t> dims(void) const { return std::vector<size_t>{}; }

	//! guess shape from the data based on the current shape
	tensorshape guess_shape(std::vector<T> data) const { return tensorshape(); }

	// >> SHAPE COMPATIBILITY <<
	//! checks if input tensor has a compatible allowed tensorshape
	bool is_same_size(const tensor<T> &other) const { return false; }

	//! check if input is compatible with tensor shape
	bool is_compatible_with(std::vector<T> data) const { return false; }

	bool is_loosely_compatible_with (std::vector<T> data) const { return false; }

	//! check if other tensor's data is compatible with this shape
	bool is_compatible_with(const tensor<T> &other) const { return false; }

	//! checks if tensorshape is aligned
	//! same number of column for each row
	bool is_aligned(void) const { return true; }

	// >>> DATA INFORMATION <<<
	//! checks if memory is allocated
	bool is_alloc(void) const { return false; }

	//! get bytes allocated
	size_t total_bytes(void) const { return 0; }

	//! get data at coordinate specified
	virtual T get(std::vector<size_t> coord) const { return T(0); }

	std::vector<T> expose(void) const { return std::vector<T>{}; }

	// >>>> MUTATOR <<<<
	//! set a new shape
	//! chop raw data outside of new shape
	//! worst case runtime: O(min(N, M))
	//! where N is the original shape size
	//! and M is the resulting shape size
	void set_shape(tensorshape shape) {}

	//! allocate raw data using allowed shape
	//! return true if successful
	bool allocate(void) { return true; }

	//! allocate raw data using input shape
	//! if shape is compatible with allowed
	//! else return false
	bool allocate(const tensorshape shape) { return true; }

	//! copy raw_data from other expanded/compressed to input shape
	bool copy_from(const tensor &other, const tensorshape shape) { return true; }

protected:
	T* raw_data_;
	tensorshape alloc_shape_;
};


// randomly initiates raw data on construction
class mock_tensor : public tensor<double>
{
public:
	mock_tensor (void) :
		tensor<double>() { randinit(); }
	mock_tensor (double scalar) :
		tensor<double>(scalar) { randinit(); }
	mock_tensor (tensorshape shape) :
		tensor<double>(shape) { randinit(); }
	mock_tensor* clone (void) const { return new mock_tensor(*this); }
	mock_tensor (const mock_tensor& other) :
		tensor<double>(other) { randinit(); }
	mock_tensor (mock_tensor&& other) :
		tensor<double>(other) { randinit(); }
	mock_tensor& operator = (const mock_tensor& other) { tensor<double>::operator=(other); return *this; }
	mock_tensor& operator = (mock_tensor&& other) { tensor<double>::operator=(other); return *this; }

	// checks if two tensors are equal without exposing
	bool equal (const mock_tensor& other) const
	{
		// check shape equality
		if (false == is_alloc() ||
			false == other.is_alloc())
		{
			return tensorshape_equal(get_shape(), other.get_shape());
		}
		if (false == alloc_shape_.is_compatible_with(other.alloc_shape_))
		{
			return false;
		}

		// check
		size_t n = alloc_shape_.n_elems();
		// crashes if we have shape, data inconsistency,
		// assuming address sanitation works properly
		return std::equal(raw_data_, raw_data_ + n, other.raw_data_);
	}

	// checks if alloc_shape_ is undefined when not allocated
	bool clean (void) const
	{
		// checks by ensuring data is null and alloc is undefined when unallocated
		return is_alloc() || (
			nullptr == raw_data_ &&
			false == alloc_shape_.is_part_defined());
	}

	double* rawptr (void) const { return raw_data_; }

	bool allocshape_is (const tensorshape& shape) { return tensorshape_equal(alloc_shape_, shape); }

protected:
	void randinit (void)
	{
		if (is_alloc())
		{
			size_t n = alloc_shape_.n_elems();
			std::vector<double> v = FUZZ<double>::get(n);
			std::memcpy(raw_data_, &v[0], sizeof(double) * n);
		}
	}
};


#endif //TENNCOR_MOCK_TENSOR_H
