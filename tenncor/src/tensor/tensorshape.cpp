//
//  tensorshape.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "tensor/tensor.hpp"

#ifdef tensorshape_hpp

namespace nnet
{

dimension dimension::merge_with (const dimension& other) const
{
	if (value_ == other.value_)
	{
		return dimension(value_);
	}
	if (value_ && other.value_)
	{
		throw std::logic_error("values do not match");
	}
	return dimension(value_ + other.value_);
}

bool dimension::is_compatible_with (const dimension& other) const
{
	return value_ == other.value_ || 0 == (value_ && other.value_);
}

void dimension::assert_is_compatible_with (const dimension& other) const
{
	assert(value_ == other.value_ || 0 == value_ || 0 == other.value_);
}

tensorshape::tensorshape (const std::vector<size_t>& dims)
{
	for (size_t d : dims)
	{
		dimensions_.push_back(dimension(d));
	}
}

tensorshape::tensorshape (const std::vector<dimension>& dims)
{
	dimensions_.assign(dims.begin(), dims.end());
}

tensorshape& tensorshape::operator = (const std::vector<size_t>& dims)
{
	for (size_t d : dims)
	{
		dimensions_.push_back(d);
	}
	return *this;
}

tensorshape tensorshape::merge_with (const tensorshape& other) const
{
	if (dimensions_.empty())
	{
		return other;
	}
	if (other.dimensions_.empty())
	{
		return *this;
	}
	if (dimensions_.size() != other.dimensions_.size())
	{
		throw std::logic_error(nnutils::formatter() << "shape of rank"
			<< dimensions_.size() << " is not compatible with shape of rank "
			<< other.dimensions_.size());
	}
	std::vector<dimension> ds;
	for (size_t i = 0; i < dimensions_.size(); i++)
	{
		try
		{
			ds.push_back(dimensions_[i].merge_with(other.dimensions_[i]));
		}
		catch (const std::logic_error& le)
		{
			throw le;
		}
	}
	return tensorshape(ds);
}

tensorshape tensorshape::concatenate (const tensorshape& other) const
{
	if (dimensions_.empty() || other.dimensions_.empty())
	{
		return tensorshape();
	}
	std::vector<dimension> ds = dimensions_;
	ds.insert(ds.end(), other.dimensions_.begin(), other.dimensions_.end());
	return tensorshape(ds);
}

tensorshape tensorshape::with_rank (size_t rank) const
{
	if (dimensions_.empty())
	{
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensorshape(ds);
	}
	if (rank != dimensions_.size())
	{
		throw std::logic_error(nnutils::formatter() 
			<< "shape does not have rank " << rank);
	}
	return *this;
}

tensorshape tensorshape::with_rank_at_least (size_t rank) const
{
	if (dimensions_.empty())
	{
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensorshape(ds);
	}
	if (rank > dimensions_.size())
	{
		throw std::logic_error(nnutils::formatter() 
			<< "shape does not have rank at least " << rank);
	}
	return *this;
}

tensorshape tensorshape::with_rank_at_most (size_t rank) const
{
	if (dimensions_.empty())
	{
		std::vector<dimension> ds;
		ds.insert(ds.end(), rank, dimension(0));
		return tensorshape(ds);
	}
	if (rank <dimensions_.size())
	{
		throw std::logic_error(nnutils::formatter() 
			<< "shape does not have rank at most " << rank);
	}
	return *this;
}

size_t tensorshape::n_elems (void) const
{
	size_t nelem = 1;
	for (dimension d : dimensions_)
	{
		nelem *= d.value_;
	}
	return nelem;
}

size_t tensorshape::n_dims (void) const
{
	return dimensions_.size();
}

std::vector<dimension> tensorshape::dims (void) const
{
	return dimensions_;
}

std::vector<size_t> tensorshape::as_list (void) const
{
	std::vector<size_t> v;
	for (dimension d : dimensions_)
	{
		v.push_back(d.value_);
	}
	return v;
}

tensorshape tensorshape::trim (void) const
{
	std::vector<dimension>::const_iterator start = dimensions_.begin();
	std::vector<dimension>::const_iterator end = --dimensions_.end();
	while (1 == size_t(*start)) { start++; }
	while (1 == size_t(*end)) { end--; }
	return std::vector<size_t>(start, end);
}

// tensorshape_proto* tensorshape::as_proto (void) const;

bool tensorshape::is_compatible_with (const tensorshape& other) const
{
	bool incap = true;
	if (!dimensions_.empty() && !other.dimensions_.empty())
	{
		if (other.dimensions_.size() == dimensions_.size())
		{
			for (size_t i = 0; i < dimensions_.size(); i++)
			{
				incap = incap && other.dimensions_[i].is_compatible_with(dimensions_[i]);
			}
		}
		else
		{
			incap = false;
		}
	}
	return incap;
}

bool tensorshape::is_part_defined (void) const
{
	return !dimensions_.empty();
}

bool tensorshape::is_fully_defined (void) const
{
	if (dimensions_.empty())
	{
		return false;
	}
	bool known = true;
	for (dimension d : dimensions_)
	{
		known = known && 0 < size_t(d);
	}
	return known;
}

void tensorshape::assert_has_rank (size_t rank) const
{
	assert(dimensions_.empty() || rank == dimensions_.size());
}

void tensorshape::assert_same_rank (const tensorshape& other) const
{
	assert(dimensions_.empty() || other.dimensions_.empty() || other.dimensions_.size() == dimensions_.size());
}

void tensorshape::assert_is_fully_defined (void) const { assert(is_fully_defined()); }

void print_shape(tensorshape ts)
{
	std::vector<size_t> shape = ts.as_list();
	if (shape.empty()) std::cout << "undefined\n";
	for (size_t dim : shape)
	{
		std::cout << dim << " ";
	}
	std::cout << std::endl;
}

}

#endif
