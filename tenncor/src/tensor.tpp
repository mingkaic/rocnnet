//
//  tensor.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <functional> // for std::bad_function_call();

#ifdef tensor_hpp

namespace nnet {

template<typename T>
void tensor<T>::copy(const tensor <T> &other) {
	if (nullptr == other.alloc) {
		alloc = nullptr;
	} else {
		alloc = other.alloc->clone();
	}
	size_t nelem = other.n_elems();
	allowed_shape = other.allowed_shape;
	alloc_shape = other.alloc_shape;

	// modify depending on raw_data implementation
	if (other.is_alloc()) {
		raw_data = alloc->allocate<T>(nelem, alloc_attrib());
	}
	memcpy(raw_data, other.raw_data, sizeof(T) * nelem);
}

template<typename T>
tensor<T>::tensor(void) {
	allowed_shape = tensor_shape(std::vector<dimension>());
}

template<typename T>
tensor<T>::tensor(const tensor_shape &shape)
		: allowed_shape(shape) {}

template<typename T>
tensor<T>::tensor(iallocator &a,
				  tensor_shape const &shape,
				  alloc_attrib const &attrib)
		: allowed_shape(shape), alloc() {
	shape.assert_is_fully_defined();
	this->alloc_shape = shape;
	alloc = a.clone();
	this->raw_data = alloc->allocate<T>(this->alloc_shape.n_elems(), attrib);
}

// tensor (operation op, size_t v_idx);

template<typename T>
tensor<T>::tensor(const tensor <T> &other) {
	copy(other);
}

template<typename T>
tensor<T>::~tensor(void) {
	// dealloc BUFFER<T> raw_data if needed
	if (nullptr != alloc) {
		alloc->dealloc(raw_data, this->alloc_shape.n_elems());
		delete alloc;
	}
}

template<typename T>
tensor <T> &tensor<T>::operator=(const tensor <T> &other) {
	if (this != &other) {
		if (nullptr != alloc) {
			alloc->dealloc(raw_data, this->alloc_shape.n_elems());
		}
		copy(other);
	}
	return *this;
}

template<typename T>
void tensor<T>::allocate(iallocator &allocer) {
	allocate(allocer, default_attr);
}

template<typename T>
void tensor<T>::allocate(iallocator &allocer, const alloc_attrib &attrib) {
	allowed_shape.assert_is_fully_defined();
	allocate(allocer, allowed_shape, attrib);
}

template<typename T>
void tensor<T>::allocate(iallocator &allocer, const tensor_shape &shape) {
	allocate(allocer, shape, default_attr);
}

template<typename T>
void tensor<T>::allocate(iallocator &allocer, const tensor_shape &shape,
						 alloc_attrib const &attrib) {
	assert(shape.is_compatible_with(allowed_shape));
	shape.assert_is_fully_defined();

	// dealloc before reallocation
	if (true == is_alloc()) {
		alloc->dealloc(raw_data, this->alloc_shape.n_elems());
	}

	alloc_shape = shape;
	alloc = allocer.clone();
	raw_data = alloc->allocate<T>(this->alloc_shape.n_elems(), attrib);
}

template<typename T>
tensor_shape tensor<T>::get_shape(void) const {
	if (is_alloc()) {
		return alloc_shape;
	}
	return allowed_shape;
}

template<typename T>
std::vector<size_t> tensor<T>::dims(void) const {
	if (is_alloc()) {
		return alloc_shape.as_list();
	}
	return allowed_shape.as_list();
}

template<typename T>
size_t tensor<T>::n_dims(void) const { return allowed_shape.n_dims(); }

template<typename T>
size_t tensor<T>::n_elems(void) const {
	if (nullptr == raw_data) {
		return 0;
	}
	return this->alloc_shape.n_elems();
}

// always aligned... until I add unaligned tensorshape
// extend tensorshape which is always aligned
template<typename T>
bool tensor<T>::is_aligned(void) const {
	return true;
}

template<typename T>
bool tensor<T>::is_same_size(const tensor <T> &other) const {
	return (this->is_alloc() && other.is_alloc() &&
			alloc_shape.is_compatible_with(other.alloc_shape)) ||
		   (this->allowed_shape.is_compatible_with(other.allowed_shape));
}

template<typename T>
bool tensor<T>::is_alloc(void) const {
	return alloc_shape.is_fully_defined() && raw_data != nullptr;
}

template<typename T>
size_t tensor<T>::total_bytes(void) const {
	return n_elems() * sizeof(T);
}

// extension of matrix index representation idx = x+y*col
template<typename T>
T tensor<T>::get(std::vector<size_t> indices) const {
	assert(is_alloc()); // otherwise meaningless
	size_t rank = alloc_shape.n_dims();
	if (indices.size() > rank) {
		throw std::logic_error(
			nnutils::formatter() << "eliciting extraneous dimensions from a tensor of rank " << rank);
	}
	std::vector<size_t> dims = alloc_shape.as_list();
	size_t accum = 1;
	size_t raw_idx = 0;
	for (size_t i = 0; i < indices.size(); i++) {
		raw_idx += indices[i] * accum;
		accum *= dims[i];
	}
	return raw_data[raw_idx];
}

template<typename T>
void tensor<T>::set_shape(tensor_shape shape) {
	assert(false == is_alloc()); // impacts allocated raw data
	this->allowed_shape = shape;
}

// how to handle shape expansion / compression?
template<typename T>
bool tensor<T>::copy_from(const tensor <T> &other, const tensor_shape &shape) {
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return false;
}

// slice along the first dimension
template<typename T>
tensor <T> tensor<T>::slice(size_t dim_start, size_t limit) {
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return tensor<T>();
}

// bool shares_buffer_with (const tensor<T>& other) const;

// template <typename T>
// size_t tensor<T>::buffer_hash (void) const {
//     return 0;
// }

// bool from_proto (const tensorproto& other);

// bool from_proto (iallocator* a, const tensorproto& other);

}

#endif
