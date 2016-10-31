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
	if (nullptr == other._alloc) {
		_alloc = nullptr;
	} else {
		_alloc = other._alloc;
	}
	size_t nelem = other.n_elems();
	_allowed_shape = other._allowed_shape;
	_alloc_shape = other._alloc_shape;

	// modify depending on _raw_data implementation
	if (other.is_alloc()) {
		_raw_data = _alloc->allocate<T>(nelem, alloc_attrib());
	}
	memcpy(_raw_data, other._raw_data, sizeof(T) * nelem);
}

template <typename T>
tensor<T>::tensor (void) : tensor(std::vector<size_t>()) {}

template<typename T>
tensor<T>::tensor (const tensor_shape shape) : _allowed_shape(shape) {
	if (_allowed_shape.is_fully_defined()) {
		_alloc_shape = shape;
		_alloc = std::make_shared<memory_alloc>();
		this->_raw_data = _alloc->allocate<T>(this->_alloc_shape.n_elems(), default_attr);
	}
}

template<typename T>
tensor<T>::tensor (std::shared_ptr<iallocator> _alloc, const tensor_shape shape) :
		tensor(_alloc, shape, default_attr) {}

template<typename T>
tensor<T>::tensor (std::shared_ptr<iallocator> _alloc,
				   const tensor_shape shape,
				   const alloc_attrib& attrib) :
		_allowed_shape(shape), _alloc_shape(shape), _alloc(_alloc) {
	shape.assert_is_fully_defined();
	this->_raw_data = _alloc->allocate<T>(this->_alloc_shape.n_elems(), attrib);
}

template<typename T>
tensor<T>::tensor (T scalar) :
		tensor(std::vector<size_t>{1}) {
	this->_raw_data[0] = scalar;
}

template<typename T>
tensor<T>::tensor (const tensor <T> &other) {
	copy(other);
}

template<typename T>
tensor<T>::~tensor (void) {
	// dealloc BUFFER<T> _raw_data if needed
	if (nullptr != _alloc) {
		_alloc->dealloc(_raw_data, this->_alloc_shape.n_elems());
	}
}

template<typename T>
tensor <T> &tensor<T>::operator = (const tensor <T> &other) {
	if (this != &other) {
		if (nullptr != _alloc) {
			_alloc->dealloc(_raw_data, this->_alloc_shape.n_elems());
		}
		copy(other);
	}
	return *this;
}

template<typename T>
void tensor<T>::allocate (std::shared_ptr<iallocator> allocer) {
	allocate(allocer, default_attr);
}

template<typename T>
void tensor<T>::allocate (std::shared_ptr<iallocator> allocer, const alloc_attrib& attrib) {
	_allowed_shape.assert_is_fully_defined();
	allocate(allocer, _allowed_shape, attrib);
}

template<typename T>
void tensor<T>::allocate (std::shared_ptr<iallocator> allocer, const tensor_shape shape) {
	allocate(allocer, shape, default_attr);
}

template<typename T>
void tensor<T>::allocate (
		std::shared_ptr<iallocator> allocer,
		const tensor_shape shape,
		const alloc_attrib& attrib) {
	assert(shape.is_compatible_with(_allowed_shape));
	shape.assert_is_fully_defined();

	// dealloc before reallocation
	if (true == is_alloc()) {
		_alloc->dealloc(_raw_data, this->_alloc_shape.n_elems());
	}

	_alloc_shape = shape;
	_alloc = allocer;
	_raw_data = _alloc->allocate<T>(this->_alloc_shape.n_elems(), attrib);
}

template<typename T>
tensor_shape tensor<T>::get_shape (void) const {
	if (is_alloc()) {
		return _alloc_shape;
	}
	return _allowed_shape;
}

template<typename T>
std::vector<size_t> tensor<T>::dims (void) const {
	if (is_alloc()) {
		return _alloc_shape.as_list();
	}
	return _allowed_shape.as_list();
}

template<typename T>
size_t tensor<T>::n_dims (void) const { return _allowed_shape.n_dims(); }

template<typename T>
size_t tensor<T>::n_elems (void) const {
	if (nullptr == _raw_data) {
		return 0;
	}
	return this->_alloc_shape.n_elems();
}

// always aligned... until I add unaligned tensorshape
// extend tensorshape which is always aligned
template<typename T>
bool tensor<T>::is_aligned (void) const {
	return true;
}

template <typename T>
tensor_shape tensor<T>::guess_shape (std::vector<T> data) const {
	if (_allowed_shape.is_fully_defined()) {
		return _allowed_shape;
	}
	std::vector<size_t> my_shape = _allowed_shape.as_list();
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
	std::vector<size_t> my_shape = _allowed_shape.as_list();
	if (is_alloc()) {
		my_shape = _alloc_shape.as_list();
	}
	size_t fixed = 1;
	for (size_t s : my_shape) {
		if (s) fixed *= s;
	}
	// not part defined means fully undefined, undefined is compatible with any data type
	return !_allowed_shape.is_part_defined() || 0 == data.size() % fixed;
}

template <typename T>
bool tensor<T>::is_compatible_with (const tensor<T>& other) const {
	return _allowed_shape.is_compatible_with(other.get_shape());
}

template<typename T>
bool tensor<T>::is_same_size (const tensor <T> &other) const {
	return (this->is_alloc() && other.is_alloc() &&
			_alloc_shape.is_compatible_with(other._alloc_shape)) ||
		   (this->_allowed_shape.is_compatible_with(other._allowed_shape));
}

template<typename T>
bool tensor<T>::is_alloc (void) const {
	return _alloc_shape.is_fully_defined() && _raw_data != nullptr;
}

template<typename T>
size_t tensor<T>::total_bytes (void) const {
	return n_elems() * sizeof(T);
}

// extension of matrix index representation idx = x+y*col
template<typename T>
T tensor<T>::get (std::vector<size_t> indices) const {
	assert(is_alloc()); // otherwise meaningless
	size_t rank = _alloc_shape.n_dims();
	if (indices.size() > rank) {
		throw std::logic_error(
			nnutils::formatter() << "eliciting extraneous dimensions from a tensor of rank " << rank);
	}
	std::vector<size_t> dims = _alloc_shape.as_list();
	size_t accum = 1;
	size_t raw_idx = 0;
	for (size_t i = 0; i < indices.size(); i++) {
		raw_idx += indices[i] * accum;
		accum *= dims[i];
	}
	return _raw_data[raw_idx];
}

template<typename T>
void tensor<T>::set_shape (tensor_shape shape) {
	assert(false == is_alloc()); // impacts allocated raw data
	this->_allowed_shape = shape;
}

// how to handle shape expansion / compression?
template<typename T>
bool tensor<T>::copy_from (const tensor<T>& other, const tensor_shape shape) {
	throw std::bad_function_call(); // NOT IMPLEMENTED
	return false;
}

// slice along the first dimension
template<typename T>
tensor <T> tensor<T>::slice (size_t dim_start, size_t limit) {
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
