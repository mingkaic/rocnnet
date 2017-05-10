//
//  tensorshape.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "tensor/tensorshape.hpp"

#ifdef TENNCOR_TENSORSHAPE_HPP

namespace nnet
{

tensorshape::tensorshape (const std::vector<size_t>& dims) :
	dimensions_(dims) {}

tensorshape& tensorshape::operator = (const std::vector<size_t>& dims)
{
	dimensions_ = dims;
	return *this;
}

std::vector<size_t> tensorshape::as_list (void) const
{
	std::vector<size_t> sampleout = dimensions_;
	if (dim_group_)
	{
		sampleout[(*dim_group_).first] = (*dim_group_).second;
	}
	return sampleout;
}

size_t tensorshape::n_elems (void) const
{
	if (dimensions_.empty())
	{
		return 0;
	}
	size_t elems = std::accumulate(dimensions_.begin(), dimensions_.end(),
	(size_t) 1, std::multiplies<size_t>());
	if (dim_group_)
	{
		elems *= (*dim_group_).second;
	}
	return elems;
}

size_t tensorshape::n_known (void) const
{
	if (dimensions_.empty())
	{
		return 0;
	}
	size_t elems = std::accumulate(dimensions_.begin(), dimensions_.end(),
	(size_t) 1,
	[](size_t a, size_t b) {
		if (b != 0)
		{
			return a * b;
		}
		return a;
	});
	if (dim_group_)
	{
		elems *= (*dim_group_).second;
	}
	return elems;
}

size_t tensorshape::rank (void) const
{
	return dimensions_.size();
}

bool tensorshape::is_compatible_with (const tensorshape& other) const
{
	bool incap = true;
	if (!dimensions_.empty() && !other.dimensions_.empty())
	{
		if (other.dimensions_.size() == dimensions_.size())
		{
			for (size_t i = 0; i < dimensions_.size(); i++)
			{
				incap = incap &&
					(dimensions_[i] == other.dimensions_[i] ||
					0 == (dimensions_[i] && other.dimensions_[i]));
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
	for (size_t d : dimensions_)
	{
		known = known && (0 < d);
	}
	return known;
}

void tensorshape::assert_has_rank (size_t rank) const
{
	assert(dimensions_.empty() || rank == dimensions_.size());
}

void tensorshape::assert_same_rank (const tensorshape& other) const
{
	assert(dimensions_.empty() || other.dimensions_.empty() ||
		other.dimensions_.size() == dimensions_.size());
}

void tensorshape::assert_is_fully_defined (void) const
{
	assert(is_fully_defined());
}

void tensorshape::undefine (void) { dimensions_.clear(); }

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
		throw std::logic_error(nnutils::formatter() << "shape of rank "
			<< dimensions_.size() << " is not compatible with shape of rank "
			<< other.dimensions_.size());
	}
	std::vector<size_t> ds;
	for (size_t i = 0; i < dimensions_.size(); i++)
	{
		size_t value = dimensions_[i];
		size_t ovalue = other.dimensions_[i];
		if (value == ovalue || (value && ovalue))
		{
			ds.push_back(value);
		}
		else
		{
			// one of the values is zero, return the non-zero value
			ds.push_back(value + ovalue);
		}
	}
	return ds;
}

tensorshape tensorshape::trim (void) const
{
	std::vector<size_t> res;
	if (false == dimensions_.empty())
	{
		size_t start = 0;
		size_t end = dimensions_.size() - 1;
		while (start < end && 1 == dimensions_.at(start)) { start++; }
		while (start < end && 1 == dimensions_.at(end)) { end--; }
		if (start < end || 1 != dimensions_.at(end))
		{
			res.insert(res.end(),
				dimensions_.begin()+start,
				dimensions_.begin()+end+1);
		}
	}
	return res;
}

tensorshape tensorshape::concatenate (const tensorshape& other) const
{
	if (dimensions_.empty())
	{
		return other;
	}
	if (other.dimensions_.empty())
	{
		return *this;
	}
	std::vector<size_t> ds = dimensions_;
	ds.insert(ds.end(), other.dimensions_.begin(), other.dimensions_.end());
	return tensorshape(ds);
}

tensorshape tensorshape::with_rank (size_t rank) const
{
	size_t ndim = dimensions_.size();
	std::vector<size_t> ds;
	if (rank < ndim)
	{
		// clip to rank
		auto it = dimensions_.begin();
		ds.insert(ds.end(), it, it+rank);
	}
	else if (rank > ndim)
	{
		// pad to fit rank
		ds = dimensions_;
		size_t diff = rank - ndim;
		ds.insert(ds.end(), diff, 1);
	}
	else
	{
		ds = dimensions_;
	}
	return ds;
}

tensorshape tensorshape::with_rank_at_least (size_t rank) const
{
	size_t ndim = dimensions_.size();
	std::vector<size_t> ds = dimensions_;
	if (rank > ndim)
	{
		// pad to fit rank
		size_t diff = rank - ndim;
		ds.insert(ds.end(), diff, 1);
	}
	return ds;
}

tensorshape tensorshape::with_rank_at_most (size_t rank) const
{
	std::vector<size_t> ds;
	if (rank < dimensions_.size())
	{
		// clip to fit rank
		auto it = dimensions_.begin();
		ds.insert(ds.end(), it, it+rank);
	}
	else
	{
		ds = dimensions_;
	}
	return ds;
}

void print_shape (tensorshape ts, std::ostream& os)
{
	std::vector<size_t> shape = ts.as_list();
	if (shape.empty())
	{
		os << "undefined";
	}
	else
	{
		for (size_t dim : shape)
		{
			os << dim << " ";
		}
	}
}

size_t tensorshape::sequential_idx (std::vector<size_t> coord) const
{
	size_t n = std::min(dimensions_.size(), coord.size());
	size_t index = 0;
	for (size_t i = 1; i < n; i++)
	{
		index += coord[n-i];
		index *= dimensions_[n-i-1];
	}
	return index + coord[0];
}

std::vector<size_t> tensorshape::coordinate_from_idx (size_t idx) const
{
	std::vector<size_t> coord;
	size_t i = idx;
	for (size_t d : dimensions_)
	{
		size_t xd = i % d;
		coord.push_back(xd);
		i = (i - xd) / d;
	}
	return coord;
}

std::vector<size_t> tensorshape::memory_indices (size_t shapeidx) const
{
	std::vector<size_t> outcoord;
	if (dim_group_)
	{
		size_t dim = (*dim_group_).first;
		size_t groupsize = (*dim_group_).second;
		std::vector<size_t> shapecoord = coordinate_from_idx(shapeidx);
		size_t idx = 0;
		size_t multiplier = 1;
		for (size_t i = 0; i < dim; i++)
		{
			multiplier *= dimensions_[i];
			idx += multiplier * shapecoord[i];
		}
		size_t lowermultiplier = multiplier;
		multiplier *= groupsize;
		for (size_t i = dim+1, n = dimensions_.size(); i < n; i++)
		{
			multiplier *= dimensions_[i];
			idx += multiplier * shapecoord[i];
		}
		// upper + lower = memory index with memory coord at dim = 0
		for (size_t i = 0; i < groupsize; i++)
		{
			outcoord.push_back(idx + i * lowermultiplier);
		}
	}
	else
	{
		outcoord = {shapeidx};
	}
	return outcoord;
}

std::vector<size_t> tensorshape::shape_dimensions (void) const
{
	return dimensions_;
}

bool tensorshape::is_grouped (void) const
{
	return (bool)dim_group_;
}

void tensorshape::group_dim (size_t dim)
{
	if (dim >= dimensions_.size()) return;
	size_t dimvalue = dimensions_[dim];
	if (dimvalue < 2) return;
	dim_group_ = std::pair<size_t,size_t>{dim, dimvalue};
	dimensions_[dim] = 1;
}

}

#endif
