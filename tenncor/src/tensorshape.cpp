//
//  tensorshape.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/tensor.hpp"

#ifdef tensorshape_hpp

namespace nnet {

dimension dimension::merge_with (const dimension& other) const {
	if (_value == other._value) {
		return dimension(_value);
	} else {
		if (_value && other._value) {
			throw std::logic_error("values do not match");
		}
		return dimension(_value + other._value);
	}
}

bool dimension::is_compatible_with (const dimension& other) const {
	return _value == other._value || 0 == (_value && other._value);
}

void dimension::assert_is_compatible_with (const dimension& other) const {
	assert(_value == other._value || 0 == _value || 0 == other._value);
}

tensor_shape::tensor_shape (const std::vector<size_t>& dims) {
	for (size_t d : dims) {
		_dimensions.push_back(dimension(d));
	}
}

tensor_shape::tensor_shape (const std::vector<dimension>& dims) {
	_dimensions.assign(dims.begin(), dims.end());
}

tensor_shape& tensor_shape::operator = (const std::vector<size_t>& dims) {
	for (size_t d : dims) {
		_dimensions.push_back(d);
	}
	return *this;
}

tensor_shape tensor_shape::merge_with (const tensor_shape& other) {
	if (_dimensions.empty()) {
		return other;
	}
	if (other._dimensions.empty()) {
		return *this;
	}
	if (_dimensions.size() != other._dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape of rank"
			<< _dimensions.size() << " is not compatible with shape of rank "
			<< other._dimensions.size());
	}
	std::vector<dimension> ds;
	for (size_t i = 0; i < _dimensions.size(); i++) {
		try {
			ds.push_back(_dimensions[i].merge_with(other._dimensions[i]));
		} catch (const std::logic_error& le) {
			throw le;
		}
	}
	return tensor_shape(ds);
}

tensor_shape tensor_shape::concatenate (const tensor_shape& other) {
	if (_dimensions.empty() || other._dimensions.empty()) {
		return tensor_shape();
	}
	std::vector<dimension> ds = _dimensions;
	ds.insert(ds.end(), other._dimensions.begin(), other._dimensions.end());
	return tensor_shape(ds);
}

tensor_shape tensor_shape::with_rank (size_t rank) {
	if (_dimensions.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank != _dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank " << rank);
	}
	return *this;
}

tensor_shape tensor_shape::with_rank_at_least (size_t rank) {
	if (_dimensions.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank > _dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank at least " << rank);
	}
	return *this;
}

tensor_shape tensor_shape::with_rank_at_most (size_t rank) {
	if (_dimensions.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank < _dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank at most " << rank);
	}
	return *this;
}

size_t tensor_shape::n_elems (void) const {
	size_t nelem = 1;
	for (dimension d : _dimensions) {
		nelem *= d._value;
	}
	return nelem;
}

size_t tensor_shape::n_dims (void) const {
	return _dimensions.size();
}

std::vector<dimension> tensor_shape::dims (void) const {
	return _dimensions;
}

std::vector<size_t> tensor_shape::as_list (void) const {
	std::vector<size_t> v;
	for (dimension d : _dimensions) {
		v.push_back(d._value);
	}
	return v;
}

// tensor_shape_proto* tensor_shape::as_proto (void) const;

bool tensor_shape::is_compatible_with (const tensor_shape& other) const {
	bool incap = true;
	if (!_dimensions.empty() && !other._dimensions.empty()) {
		if (other._dimensions.size() == _dimensions.size()) {
			for (size_t i = 0; i < _dimensions.size(); i++) {
				incap = incap && other._dimensions[i].is_compatible_with(_dimensions[i]);
			}
		} else {
			incap = false;
		}
	}
	return incap;
}

bool tensor_shape::is_part_defined (void) const {
	return !_dimensions.empty();
}

bool tensor_shape::is_fully_defined (void) const {
	if (_dimensions.empty()) {
		return false;
	}
	bool known = true;
	for (dimension d : _dimensions) {
		known = known && 0 < size_t(d);
	}
	return known;
}

void tensor_shape::assert_has_rank (size_t rank) const {
	assert(_dimensions.empty() || rank == _dimensions.size());
}

void tensor_shape::assert_same_rank (const tensor_shape& other) const {
	assert(_dimensions.empty() || other._dimensions.empty() || other._dimensions.size() == _dimensions.size());
}

}

#endif
