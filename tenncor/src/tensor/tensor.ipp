//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <functional> // for std::bad_function_call();

#ifdef tensor_hpp

namespace nnet {

template<typename T>
void tensor<T>::copy (const tensor <T> &other) {
	if (nullptr == other.alloc_) {
		alloc_ = nullptr;
	} else {
		alloc_ = other.alloc_;
	}
	size_t nelem = other.n_elems();
	allowed_shape_ = other.allowed_shape_;
	alloc_shape_ = other.alloc_shape_;

	// modify depending on raw_data_ implementation
	if (other.is_alloc()) {
		raw_data_ = alloc_->allocate<T>(nelem, alloc_attrib());
	}
	memcpy(raw_data_, other.raw_data_, sizeof(T) * nelem);
}

template <typename T>
tensor<T>::tensor (void) : tensor<T>(std::vector<size_t>()) {}

template<typename T>
tensor<T>::tensor (tensorshape shape) : allowed_shape_(shape) {
	if (allowed_shape_.is_fully_defined()) {
		alloc_shape_ = shape;
		alloc_ = new ram_alloc();
		this->raw_data_ = alloc_->allocate<T>(this->alloc_shape_.n_elems(), default_attr);
	}
}

template<typename T>
tensor<T>::tensor (tensorshape shape, iallocator& alloc) :
	tensor<T>(shape, alloc, default_attr) {}

template<typename T>
tensor<T>::tensor (tensorshape shape, iallocator* alloc) :
	tensor<T>(shape, *alloc, default_attr) { delete alloc; }

template<typename T>
tensor<T>::tensor (tensorshape shape, iallocator& alloc, const alloc_attrib& attrib) :
		allowed_shape_(shape), alloc_shape_(shape), alloc_(alloc.clone()) {
	if (shape.is_fully_defined()) {
		this->raw_data_ = alloc_->allocate<T>(this->alloc_shape_.n_elems(), attrib);
	}
}

template<typename T>
tensor<T>::tensor (tensorshape shape, iallocator* alloc, const alloc_attrib& attrib) :
	tensor<T>(shape, *alloc, attrib) { delete alloc; }

template<typename T>
tensor<T>::tensor (T scalar) : tensor<T>(std::vector<size_t>{1}) {
	this->raw_data_[0] = scalar;
}

template<typename T>
tensor<T>::tensor (const tensor<T>& other) {
	this->copy(other);
}

template<typename T>
tensor<T>::~tensor (void) {
	// dealloc BUFFER<T> raw_data_ if needed
	if (nullptr != alloc_) {
		alloc_->dealloc(raw_data_, this->alloc_shape_.n_elems());
		delete alloc_;
	}
}

template<typename T>
tensor<T>& tensor<T>::operator = (const tensor<T>& other) {
	if (this != &other) {
		if (nullptr != alloc_) {
			alloc_->dealloc(raw_data_, this->alloc_shape_.n_elems());
		}
		this->copy(other);
	}
	return *this;
}

template<typename T>
void tensor<T>::allocate (iallocator* allocer) {
	allocate(allocer, default_attr);
}

template<typename T>
void tensor<T>::allocate (iallocator* allocer, const alloc_attrib& attrib) {
	allowed_shape_.assert_is_fully_defined();
	allocate(allocer, allowed_shape_, attrib);
}

template<typename T>
void tensor<T>::allocate (iallocator* allocer, const tensorshape shape) {
	allocate(allocer, shape, default_attr);
}

template<typename T>
void tensor<T>::allocate (iallocator* allocer,
		const tensorshape shape, const alloc_attrib& attrib) {
	assert(shape.is_compatible_with(allowed_shape_));
	shape.assert_is_fully_defined();

	// dealloc before reallocation
	if (true == is_alloc()) {
		alloc_->dealloc(raw_data_, this->alloc_shape_.n_elems());
	}

	alloc_shape_ = shape;
	alloc_ = allocer;
	raw_data_ = alloc_->allocate<T>(this->alloc_shape_.n_elems(), attrib);
}

template<typename T>
tensorshape tensor<T>::get_shape (void) const {
	if (is_alloc()) {
		return alloc_shape_;
	}
	return allowed_shape_;
}

template<typename T>
std::vector<size_t> tensor<T>::dims (void) const {
	if (is_alloc()) {
		return alloc_shape_.as_list();
	}
	return allowed_shape_.as_list();
}

template<typename T>
size_t tensor<T>::n_dims (void) const { return allowed_shape_.n_dims(); }

template<typename T>
size_t tensor<T>::n_elems (void) const {
	if (nullptr == raw_data_) {
		return 0;
	}
	return this->alloc_shape_.n_elems();
}

// always aligned... until I add unaligned tensorshape
// extend tensorshape which is always aligned
template<typename T>
bool tensor<T>::is_aligned (void) const {
	return true;
}

template <typename T>
tensorshape tensor<T>::guess_shape (std::vector<T> data) const {
	if (allowed_shape_.is_fully_defined()) {
		return allowed_shape_;
	}
	std::vector<size_t> my_shape = allowed_shape_.as_list();
	size_t first_undef = my_shape.size();
	size_t fixed = 1;
	for (size_t i = 0; i < my_shape.size(); i++) {
		if (0 == my_shape[i]) {
			if (first_undef > i) first_undef = i;
			my_shape[i] = 1;
		} else {
			fixed *= my_shape[i];
		}
	}
	my_shape[first_undef] = data.size() / fixed;
	return my_shape;
}

template <typename T>
bool tensor<T>::is_compatible_with (std::vector<T> data) const {
	std::vector<size_t> my_shape = allowed_shape_.as_list();
	if (is_alloc()) {
		my_shape = alloc_shape_.as_list();
	}
	size_t fixed = 1;
	for (size_t s : my_shape) {
		if (s) fixed *= s;
	}
	// not part defined means fully undefined, undefined is compatible with any data type
	return !allowed_shape_.is_part_defined() || 0 == data.size() % fixed;
}

template <typename T>
bool tensor<T>::is_compatible_with (const tensor<T>& other) const {
	return allowed_shape_.is_compatible_with(other.get_shape());
}

template<typename T>
bool tensor<T>::is_same_size (const tensor<T>& other) const {
	tensorshape simp_shape = alloc_shape_.trim();
	tensorshape other_simp = other.alloc_shape_.trim();

	return (this->is_alloc() && other.is_alloc() &&
			simp_shape.is_compatible_with(other_simp)) ||
		   (this->allowed_shape_.is_compatible_with(other.allowed_shape_));
}

template<typename T>
bool tensor<T>::is_alloc (void) const {
	return alloc_shape_.is_fully_defined() && raw_data_ != nullptr;
}

template<typename T>
size_t tensor<T>::total_bytes (void) const {
	return n_elems() * sizeof(T);
}

// extension of matrix index representation idx = x+y*col
template<typename T>
T tensor<T>::get (std::vector<size_t> indices) const {
	assert(is_alloc()); // otherwise meaningless
	size_t rank = alloc_shape_.n_dims();
	if (indices.size() > rank) {
		throw std::logic_error(
			nnutils::formatter() << "eliciting extraneous dimensions from a tensor of rank " << rank);
	}
	std::vector<size_t> dims = alloc_shape_.as_list();
	size_t accum = 1;
	size_t raw_idx = 0;
	for (size_t i = 0; i < indices.size(); i++) {
		raw_idx += indices[i] * accum;
		accum *= dims[i];
	}
	return raw_data_[raw_idx];
}

template<typename T>
void tensor<T>::set_shape (tensorshape shape) {
	assert(false == is_alloc()); // impacts allocated raw data
	this->allowed_shape_ = shape;
}

// how to handle shape expansion / compression?
template<typename T>
bool tensor<T>::copy_from (const tensor<T>& other, const tensorshape shape) {
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return false;
}

// slice along the first dimension
template<typename T>
tensor<T> tensor<T>::slice (size_t dim_start, size_t limit) {
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return tensor<T>();
}

// bool shares_buffer_with (const tensor<T>& other) const;

// template <typename T>
// size_t tensor<T>::buffer_hash (void) const {
//	 return 0;
// }

// bool from_proto (const tensorproto& other);

// bool from_proto (iallocator* a, const tensorproto& other);

}

#endif
