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
	if (value_ == other.value_) {
		return dimension(value_);
	}
	if (value_ && other.value_) {
		throw std::logic_error("values do not match");
	}
	return dimension(value_ + other.value_);
}

bool dimension::is_compatible_with (const dimension& other) const {
	return value_ == other.value_ || 0 == (value_ && other.value_);
}

void dimension::assert_is_compatible_with (const dimension& other) const {
	assert(value_ == other.value_ || 0 == value_ || 0 == other.value_);
}

tensor_shape::tensor_shape (const std::vector<size_t>& dims) {
	for (size_t d : dims) {
		 dimensions_.push_back(dimension(d));
	}
}

tensor_shape::tensor_shape (const std::vector<dimension>& dims) {
	 dimensions_.assign(dims.begin(), dims.end());
}

tensor_shape& tensor_shape::operator = (const std::vector<size_t>& dims) {
	for (size_t d : dims) {
		 dimensions_.push_back(d);
	}
	return *this;
}

tensor_shape tensor_shape::merge_with (const tensor_shape& other) {
	if ( dimensions_.empty()) {
		return other;
	}
	if (other.dimensions_.empty()) {
		return *this;
	}
	if ( dimensions_.size() != other.dimensions_.size()) {
		throw std::logic_error(nnutils::formatter() << "shape of rank"
			<<  dimensions_.size() << " is not compatible with shape of rank "
			<< other.dimensions_.size());
	}
	std::vector<dimension> ds;
	for (size_t i = 0; i <  dimensions_.size(); i++) {
		try {
			ds.push_back( dimensions_[i].merge_with(other.dimensions_[i]));
		} catch (const std::logic_error& le) {
			throw le;
		}
	}
	return tensor_shape(ds);
}

tensor_shape tensor_shape::concatenate (const tensor_shape& other) {
	if ( dimensions_.empty() || other.dimensions_.empty()) {
		return tensor_shape();
	}
	std::vector<dimension> ds =  dimensions_;
	ds.insert(ds.end(), other.dimensions_.begin(), other.dimensions_.end());
	return tensor_shape(ds);
}

tensor_shape tensor_shape::with_rank (size_t rank) {
	if ( dimensions_.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank !=  dimensions_.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank " << rank);
	}
	return *this;
}

tensor_shape tensor_shape::with_rank_at_least (size_t rank) {
	if ( dimensions_.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank >  dimensions_.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank at least " << rank);
	}
	return *this;
}

tensor_shape tensor_shape::with_rank_at_most (size_t rank) {
	if ( dimensions_.empty()) {
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensor_shape(ds);
	}
	if (rank <  dimensions_.size()) {
		throw std::logic_error(nnutils::formatter() << "shape does not have rank at most " << rank);
	}
	return *this;
}

size_t tensor_shape::n_elems (void) const {
	size_t nelem = 1;
	for (dimension d :  dimensions_) {
		nelem *= d.value_;
	}
	return nelem;
}

size_t tensor_shape::n_dims (void) const {
	return dimensions_.size();
}

std::vector<dimension> tensor_shape::dims (void) const {
	return dimensions_;
}

std::vector<size_t> tensor_shape::as_list (void) const {
	std::vector<size_t> v;
	for (dimension d :  dimensions_) {
		v.push_back(d.value_);
	}
	return v;
}

tensor_shape tensor_shape::trim (void) const {
	std::vector<dimension>::const_iterator start = dimensions_.begin();
	std::vector<dimension>::const_iterator end = --dimensions_.end();
	while (1 == size_t(*start)) { start++; }
	while (1 == size_t(*end)) { end--; }
	return std::vector<size_t>(start, end);
}

// tensor_shape_proto* tensor_shape::as_proto (void) const;

bool tensor_shape::is_compatible_with (const tensor_shape& other) const {
	bool incap = true;
	if (! dimensions_.empty() && !other.dimensions_.empty()) {
		if (other.dimensions_.size() ==  dimensions_.size()) {
			for (size_t i = 0; i <  dimensions_.size(); i++) {
				incap = incap && other.dimensions_[i].is_compatible_with(dimensions_[i]);
			}
		} else {
			incap = false;
		}
	}
	return incap;
}

bool tensor_shape::is_part_defined (void) const {
	return ! dimensions_.empty();
}

bool tensor_shape::is_fully_defined (void) const {
	if ( dimensions_.empty()) {
		return false;
	}
	bool known = true;
	for (dimension d :  dimensions_) {
		known = known && 0 < size_t(d);
	}
	return known;
}

void tensor_shape::assert_has_rank (size_t rank) const {
	assert( dimensions_.empty() || rank ==  dimensions_.size());
}

void tensor_shape::assert_same_rank (const tensor_shape& other) const {
	assert( dimensions_.empty() || other.dimensions_.empty() || other.dimensions_.size() ==  dimensions_.size());
}

}

#endif
