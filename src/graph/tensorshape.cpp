//
//  tensorshape.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

#ifdef tensorshape_hpp

namespace nnet {

dimension dimension::merge_with (const dimension& other) const {
	if (value == other.value) {
		return dimension(value);
	} else {
		if (value && other.value) {
			throw std::logic_error("values do not match");
		}
		return dimension(value + other.value);
	}
}

bool dimension::is_compatible_with (const dimension& other) const {
	return value == other.value || 0 == (value && other.value);
}

void dimension::assert_is_compatible_with (const dimension& other) const {
	assert(value == other.value || 0 == value || 0 == other.value);
}

tensor_shape::tensor_shape (const std::vector<size_t>& dims) {
	for (size_t d : dims) {
		dimensions.push_back(dimension(d));
	}
}

tensor_shape::tensor_shape (const std::vector<dimension>& dims) {
	dimensions.assign(dims.begin(), dims.end());
}

tensor_shape& tensor_shape::operator = (const std::vector<size_t>& dims) {
	for (size_t d : dims) {
		dimensions.push_back(d);
	}
	return *this;
}

tensor_shape tensor_shape::merge_with (const tensor_shape& other) {
	if (dimensions.empty()) {
		return other;
	}
	if (other.dimensions.empty()) {
		return *this;
	}
	if (dimensions.size() != other.dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape of rank"
			<< dimensions.size() << " is not compatible with shape of rank "
			<< other.dimensions.size());
	}
	std::vector<dimension> ds;
	for (size_t i = 0; i < dimensions.size(); i++) {
		try {
			ds.push_back(dimensions[i].merge_with(other.dimensions[i]));
		} catch (const std::logic_error& le) {
			throw le;
		}
	}
	return tensor_shape(ds);
}

tensor_shape tensor_shape::concatenate (const tensor_shape& other) {
	if (dimensions.empty() || other.dimensions.empty()) {
		return tensor_shape();
	}
	std::vector<dimension> ds = dimensions;
	ds.insert(ds.end(), other.dimensions.begin(), other.dimensions.end());
	return tensor_shape(ds);
}

tensor_shape tensor_shape::with_rank (size_t rank) {
	if (dimensions.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank != dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank " << rank);
	}
	return *this;
}

tensor_shape tensor_shape::with_rank_at_least (size_t rank) {
	if (dimensions.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank > dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank at least " << rank);
	}
	return *this;
}

tensor_shape tensor_shape::with_rank_at_most (size_t rank) {
	if (dimensions.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank < dimensions.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank at most " << rank);
	}
	return *this;
}

size_t tensor_shape::n_elems (void) const {
	size_t nelem = 1;
	for (dimension d : dimensions) {
		nelem *= d.value;
	}
	return nelem;
}

size_t tensor_shape::n_dims (void) const {
	return dimensions.size();
}

std::vector<dimension> tensor_shape::dims (void) const {
	return dimensions;
}

std::vector<size_t> tensor_shape::as_list (void) const {
	std::vector<size_t> v;
	for (dimension d : dimensions) {
		v.push_back(d.value);
	}
	return v;
}

// tensor_shape_proto* tensor_shape::as_proto (void) const;

bool tensor_shape::is_compatible_with (const tensor_shape& other) const {
	bool incap = true;
	if (!dimensions.empty() && !other.dimensions.empty()) {
		if (other.dimensions.size() == dimensions.size()) {
			for (size_t i = 0; i < dimensions.size(); i++) {
				incap = incap && other.dimensions[i].is_compatible_with(dimensions[i]);
			}
		} else {
			incap = false;
		}
	}
	return incap;
}

bool tensor_shape::is_part_defined (void) const {
	return !dimensions.empty();
}

bool tensor_shape::is_fully_defined (void) const {
	if (dimensions.empty()) {
		return false;
	}
	bool known = true;
	for (dimension d : dimensions) {
		known = known && 0 < size_t(d);
	}
	return known;
}

void tensor_shape::assert_has_rank (size_t rank) const {
	assert(dimensions.empty() || rank == dimensions.size());
}

void tensor_shape::assert_same_rank (const tensor_shape& other) const {
	assert(dimensions.empty() || other.dimensions.empty() || other.dimensions.size() == dimensions.size());
}

}

#endif
