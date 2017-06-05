//
//  transform.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TRANSFORM_HPP

#include "graph/connector/immutable/mappable.hpp"

namespace nnet
{

template <typename T>
varptr<T> transpose (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	assert(2 >= a->get_shape().rank());
	return immutable<T>::get(std::vector<inode<T>*>{a},
	[](std::vector<tensorshape> shapes)
	{
		tensorshape ts = shapes[0];
		if (ts.is_fully_defined())
		{
			// restrict shape to no greater than 2-D for now
			assert(ts.rank() <= 2);
			std::vector<size_t> inl = ts.as_list();
			if (ts.rank() == 1)
			{
				return std::vector<size_t>{1, inl[0]};
			}
			return std::vector<size_t>{inl[1], inl[0]};
		}
		return std::vector<size_t>{};
	},
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		const T* src = args[0];
		// we have the new shape
		std::vector<size_t> outs = shape.as_list();
		// old dimensions
		size_t dimX = outs[1]; size_t dimY = outs[0];
		// not in place so x = y+1 doesn't work
		for (size_t y = 0; y < dimY; y++)
		{
			for (size_t x = 0; x < dimX; x++)
			{
				dest[y+x*dimY] = src[x+y*dimX];
			}
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* grad;
		args.front()->get_leaf(grad, leaf);
		return transpose(varptr<T>(grad));
	}, "transpose");
}

template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch)
{
	if (nullptr == a || nullptr == watch) return nullptr;
	if (a->good_status() && *a == (T)0) return a;
	// additional constraint that watch shape must be have shape with
	// dimensions greater or equal to a's dimensional value (shape.as_list()[i])
	return immutable<T>::get(std::vector<inode<T>*>{a, watch},
	[](std::vector<tensorshape> shapes)
	{
		return shapes[1]; // watch is always argument 2
	},
	[](T* dest, const tensorshape& outshape, std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
	{
		const T* src = args[0];
		tensorshape& inshape = inshapes[0];
		fit_toshape(dest, outshape, src, inshape);
	},
	[watch](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* grad;
		args.front()->get_leaf(grad, leaf);
		return grad;
	}, "fit", watch);
}

template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier)
{
	if (nullptr == a) return nullptr;
	if (multiplier == 0) return constant<T>::get(0);
	if (multiplier == 1) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a},
	[index, multiplier](std::vector<tensorshape> shapes)
	{
		tensorshape ts = shapes[0];
		ts.assert_is_fully_defined();
		std::vector<size_t> tv = ts.as_list();
		// allocated additional space along index
		size_t dims = ts.rank();
		if (index >= dims)
		{
			// extending extra dimensions
			size_t extra_dims = index - dims;
			if (extra_dims)
			{
				tv.insert(tv.end(), extra_dims, 1);
			}
			tv.push_back(multiplier);
		}
		else
		{
			tv[index] *= multiplier;
		}
		return tv;
	},
	[index, multiplier](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		const T* src = args[0];
		// REMEMBER that ts is the resulting shape, not the original shape
		// both above and below values are calculations based on the original shape
		std::vector<size_t> tv = shape.as_list();
		// below calculates all elements encompassed up to the index dimension
		// that is for a shape of <1, 2, 3, 4> and index 2
		// below = 1 * 2 * 3 = 6
		size_t below = 1;
		for (size_t i = 0; i < index; i++)
		{
			below *= tv[i];
		}
		// we know that for the resulting shape, the dimensional-value at index is multiplied by multiplier
		// so to obtain the original dimension, we divide by multiplier
		below *= tv[index] / multiplier;
		// above calculates the number of tensors (of index rank) within the original tensor
		// that is for a shape of <1, 2, 3, 4> and index 2
		// the tensors of index rank is represented by the first 3 dimensions <1, 2, 3>
		// the overall tensor is represented as a tensor of tensor < <1, 2, 3>, 4>
		// above is 4
		// above = original total / below
		// original total = resulting total / multiplier
		size_t above = shape.n_elems() / (multiplier * below);

		// copy over data
		for (size_t i = 0; i < above; i++)
		{
			// copy data multiplier times
			const T* src_addr = src + i * below;
			for (size_t j = 0; j < multiplier; j++)
			{
				T* dest_addr = dest + below * (multiplier * i + j);
				std::memcpy(dest_addr, src_addr, below * sizeof(T));
			}
		}
	},
	[index, multiplier](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* grad;
		args.front()->get_leaf(grad, leaf);
		return grad;
	}, "extend");
}

template <typename T>
varptr<T> compress (const varptr<T> a, optional<size_t> index,
	std::function<T(const std::vector<T>&)> collector)
{
	if (nullptr == a) return nullptr;
	FORWARD_OP<T> gatherer;
	SHAPER shaper;
	if (index)
	{
		gatherer =
		[index, collector](T* dest, const tensorshape&, std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
		{
			size_t idx = *index;
			const T* src = args[0];
			tensorshape& orig = inshapes[0];
			if (idx >= orig.rank())
			{
				std::memcpy(dest, src, sizeof(T)*orig.n_elems());
				return;
			}
			// REMEMBER that ts is the resulting shape, not the original shape
			// both above and below values are calculations based on the original shape
			// original shape
			std::vector<size_t> tv = orig.as_list();
			size_t idx_val = tv[idx];
			// below for compression calculates all elements below the index dimension
			// that is for a shape of <1, 2, 3, 4> and index 2
			// below = 1 * 2 = 2
			size_t below = 1;
			for (size_t i = 0; i < idx; i++)
			{
				below *= tv[i];
			}
			// above denotes the same above as the one in extend
			size_t above = orig.n_elems() / (below*idx_val);

			// copy over data
			for (size_t i = 0; i < above; i++)
			{
				for (size_t j = 0; j < below; j++)
				{
					// apply compression to each element along idx_val dimension
					size_t dest_idx = j + i * below;
					std::vector<T> gather;
					for (size_t k = 0; k < idx_val; k++)
					{
						gather.push_back(src[j + k * below + i * below * idx_val]);
					}
					dest[dest_idx] = collector(gather);
				}
			}
		};
		shaper =
		[index](std::vector<tensorshape> shapes)
		{
			size_t idx = *index;
			tensorshape& ts = shapes[0];
			ts.assert_is_fully_defined();

			size_t srank = ts.rank();
			if (idx >= srank)
			{
				return ts;
			}
			std::vector<size_t> tv = ts.as_list();
			if (1 == srank)
			{
				tv[0] = 1;
			}
			else if (0 == idx)
			{ // pop front
				tv.front() = std::move(tv.back());
				tv.pop_back();
			}
			else if (tv.size()-1 == idx)
			{
				tv.pop_back();
			}
			else
			{
				tv[idx] = 1;
			}
			return tensorshape(tv);
		};
	}
	else
	{
		gatherer =
		[collector](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
		{
			dest[0] = collector(std::vector<T>(args[0], args[0]+shape.n_elems()));
		};
		// scalar shape
		shaper = [](std::vector<tensorshape>) { return std::vector<size_t>{1}; };
	}

	return immutable<T>::get(std::vector<inode<T>*>{a}, shaper, gatherer,
	[index, collector](std::vector<inode<T>*> args, variable<T>* leaf) -> inode<T>*
	{
		inode<T>* gradn;
		args.front()->get_leaf(gradn, leaf);
		if (index)
		{
			return mappable<T>::get(gradn, *index);
		}
		return gradn;
	}, "compress");
}

template <typename T>
varptr<T> reduce_max (const varptr<T> a, optional<size_t> dimension)
{
	return compress<T>(a, dimension,
	[](const std::vector<T>& values) -> T
	{
		return *std::max_element(values.begin(), values.end());
	});
}

template <typename T>
varptr<T> reduce_sum (const varptr<T> a, optional<size_t> dimension)
{
	return compress<T>(a, dimension,
	[](const std::vector<T>& values) -> T
	{
		return std::accumulate(values.begin(), values.end(), (T)0);
	});
}

template <typename T>
varptr<T> reduce_mean (const varptr<T> a, optional<size_t> dimension)
{
	return compress<T>(a, dimension,
	[](const std::vector<T>& values) -> T
	{
		return std::accumulate(values.begin(), values.end(), (T)0) / values.size();
	});
}

template <typename T>
varptr<T> arg_compress (const varptr<T> a, optional<size_t> dimension,
	std::function<size_t(const std::vector<T>&)> search)
{
	if (nullptr == a) return nullptr;
	FORWARD_OP<T> gatherer;
	SHAPER shaper;
	if (dimension)
	{
		gatherer =
		[dimension, search](T* dest, const tensorshape&, std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
		{
			size_t dim = *dimension;
			const T* src = args[0];
			// REMEMBER that ts is the resulting shape, not the original shape
			// both above and below values are calculations based on the original shape
			// original shape
			tensorshape& orig = inshapes[0];
			if (dim >= orig.rank())
			{
				throw std::logic_error(nnutils::formatter() << "attempting to obtain arg index along dimension "
					<< dim << " on a " << orig.rank() << " tensor");
			}
			std::vector<size_t> tv = orig.as_list();
			size_t idx_val = tv[dim];
			// below for compression calculates all elements below the index dimension
			// that is for a shape of <1, 2, 3, 4> and index 2
			// below = 1 * 2 = 2
			size_t below = 1;
			for (size_t i = 0; i < dim; i++)
			{
				below *= tv[i];
			}
			// above denotes the same above as the one in extend
			size_t above = orig.n_elems() / (below*idx_val);

			// copy over data
			for (size_t i = 0; i < above; i++)
			{
				for (size_t j = 0; j < below; j++)
				{
					// apply compression to each element along idx_val dimension
					size_t dest_idx = j + i * below;
					std::vector<T> vals;
					for (size_t k = 0; k < idx_val; k++)
					{
						vals.push_back(src[j + k * below + i * below * idx_val]);
					}
					dest[dest_idx] = (T)search(vals);
				}
			}
		};
		shaper =
		[dimension](std::vector<tensorshape> shapes)
		{
			size_t dim = *dimension;
			tensorshape ts = shapes[0];
			ts.assert_is_fully_defined();
			if (dim >= ts.rank())
			{
				throw std::logic_error(nnutils::formatter() << "attempting to obtain arg index along dimension "
					<< dim << " on a " << ts.rank() << " tensor");
			}
			std::vector<size_t> tv = ts.as_list();
			tv[dim] = 1;
			if (tv.size() > 1)
			{
				if (0 == dim)
				// pop front
				{
					tv.front() = std::move(tv.back());
					tv.pop_back();
				}
				else if (tv.size()-1 == dim)
				{
					tv.pop_back();
				}
			}
			return tv;
		};
	}
	else
	{
		gatherer =
		[search](T* dest, const tensorshape&, std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
		{
			tensorshape& shape = inshapes[0];
			const T* indata = args[0];
			size_t idx = search(std::vector<T>(indata[0], indata[0]+shape.n_elems()));
			std::vector<size_t> coord = shape.coordinate_from_idx(idx);
			std::vector<T> tcoord(coord.begin(), coord.end());
			memcpy(dest, &tcoord[0], tcoord.size() * sizeof(T));
		};
		// scalar shape
		shaper = [](std::vector<tensorshape> inshapes) { return std::vector<size_t>{inshapes[0].rank()}; };
	}

	return immutable<T>::get(std::vector<inode<T>*>{a}, shaper, gatherer,
	[dimension, search](std::vector<inode<T>*>, variable<T>*)
	{
		// arg_compression's gradient has no intrinsic meaning
		throw std::logic_error("attempting to get gradient of arg compression: undefined and meaningless operation");
		return nullptr;
	}, "argcompress");
}

template <typename T>
varptr<T> arg_max (const varptr<T> a, optional<size_t> dimension)
{
	return arg_compress<T>(a, dimension,
	[](const std::vector<T>& vec) -> size_t
	{
		auto mit = std::max_element(vec.begin(), vec.end());
		return std::distance(vec.begin(), mit);
	});
}

}

#endif
