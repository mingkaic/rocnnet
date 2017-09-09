//
//  transform.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/connector/immutable/shape_dep.hpp"

#ifdef TENNCOR_TRANSFORM_HPP

namespace nnet
{

template <typename T>
varptr<T> transpose (const varptr<T> a, std::pair<size_t,size_t> axis_swap)
{
	if (nullptr == a.get()) return nullptr;
	// order the axis by min, max
	if (axis_swap.first > axis_swap.second)
	{
		std::swap(axis_swap.first, axis_swap.second);
	}
	std::string opname = nnutils::formatter() << "transpose_" << axis_swap.first << "_" << axis_swap.second;
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	return immutable<T>::get(std::vector<inode<T>*>{a},
	[axis_swap](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape ts = shapes[0];
		if (ts.is_fully_defined())
		{
			std::vector<size_t> inl = ts.as_list();
			if (axis_swap.second >= inl.size())
			{
				inl.insert(inl.end(), axis_swap.second - inl.size() + 1, 1);
			}
			std::swap(inl[axis_swap.first], inl[axis_swap.second]);
			return inl;
		}
		return tensorshape();
	},
	new transfer_func<T>(
	[axis_swap](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		size_t rank = shape.ins_[0].rank();
		if (axis_swap.first >= rank)
		{
			return;
		}
		std::vector<size_t> coords;
		if (axis_swap.second >= rank)
		{
			// we're transposing with a previously non-existent dimension,
			// so there is a 1-1 correspondence between src and dest
			std::memcpy(dest, src[0], n_elems * sizeof(T));
		}
		else
		{
			for (size_t i = 0; i < n_elems; i++)
			{
				coords = shape.outs_.coordinate_from_idx(i);
				std::swap(coords[axis_swap.first], coords[axis_swap.second]);
				dest[i] = src[0][shape.ins_[0].flat_idx(coords)];
			}
		}
	}),
	[axis_swap](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> grad = args.front().second;
		return transpose(grad, axis_swap);
	}, opname);
}

template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch)
{
	if (nullptr == a.get() || nullptr == watch.get()) return nullptr;
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	if (aconst && *aconst == (T)0) return a;
	// additional constraint that watch shape must be have shape with
	// dimensions greater or equal to a's dimensional value (shape.as_list()[i])
	std::string opname = "fit";
	if (inode<T>* parent = ordered_binary_parent_search(a.get(), watch.get(), opname))
	{
		return parent;
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, watch},
	[](std::vector<tensorshape> shapes) -> tensorshape
	{
		return shapes[1]; // watch is always argument 2
	},
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(2 == src.size());
		std::vector<size_t> src_list = shape.ins_[0].as_list();
		size_t minrank = std::min(src_list.size(), shape.ins_[1].rank());
		size_t n_elems = shape.outs_.n_elems();
		std::vector<size_t> coords;
		for (size_t i = 0; i < n_elems; i++)
		{
			coords = shape.outs_.coordinate_from_idx(i);
			bool inbound = true;
			for (size_t j = 0; inbound && j < minrank; j++)
			{
				inbound = coords[j] < src_list[j];
			}
			for (size_t j = minrank, rank = coords.size(); inbound && j < rank; j++)
			{
				inbound = coords[j] == 0;
			}
			dest[i] = inbound ? src[0][shape.ins_[0].flat_idx(coords)] : 0;
		}
	}),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> grad = args.front().second;
		return grad;
	}, opname, watch);
}

template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier)
{
	if (nullptr == a.get()) return nullptr;
	if (multiplier == 0) return constant<T>::get(0);
	if (multiplier == 1) return a;
	std::string opname = nnutils::formatter() << "extend_" << index << "_" << multiplier;
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	return immutable<T>::get(std::vector<inode<T>*>{a},
	[index, multiplier](std::vector<tensorshape> shapes) -> tensorshape
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
	new transfer_func<T>([index, multiplier](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		size_t dim = shape.ins_[0].as_list()[index];
		std::vector<size_t> coords;
		for (size_t i = 0; i < n_elems; i++)
		{
			std::vector<size_t> coords = shape.outs_.coordinate_from_idx(i);
			coords[index] = coords[index] % dim;

			dest[i] = src[0][shape.ins_[0].flat_idx(coords)];
		}
	}),
	[index, multiplier](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> grad = args.front().second;
		return grad;
	}, opname);
}

template <typename T>
varptr<T> compress (const varptr<T> a, AGGREGATE<T> collector,
	optional<size_t> index, std::string name,
	std::function<varptr<T>(varptr<T>,varptr<T>)> bprop)
{
	if (nullptr == a.get()) return nullptr;
	std::string imm_name = (bool) index ? nnutils::formatter() << name << "_" << *index : name;
	if (inode<T>* parent = unary_parent_search(a.get(), imm_name))
	{
		return parent;
	}
	SHAPER shaper;
	transfer_func<T>* forward;
	if ((bool) index)
	{
		shaper = [index](std::vector<tensorshape> shapes) -> tensorshape
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
				// pop front
			{
				tv = std::vector<size_t>(tv.begin()+1, tv.end());
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

		forward = new transfer_func<T>(
		[index, collector](T* dest, std::vector<const T*> src, shape_io shape)
		{
			assert(1 == src.size());
			size_t n_elems = shape.outs_.n_elems();
			if (shape.ins_[0].rank() <= *index)
			{
				std::memcpy(dest, src[0], sizeof(T) * n_elems);
			}
			else
			{
				size_t adim = shape.ins_[0].as_list()[*index];
				std::vector<size_t> coords;
				for (size_t i = 0; i < n_elems; i++)
				{
					coords = shape.outs_.coordinate_from_idx(i);
					if (*index == coords.size())
					{
						coords.push_back(0);
					}
					else if (*index == 0)
					{
						std::vector<size_t> temp = coords;
						coords = {0};
						coords.insert(coords.end(), temp.begin(), temp.end());
					}
					for (size_t j = 0; j < adim; j++)
					{
						coords[*index] = j;
						size_t src_idx = shape.ins_[0].flat_idx(coords);
						dest[i] = collector(dest[i], src[0][src_idx]);
					}
				}
			}
		});
	}
	else
	{
		shaper = [](std::vector<tensorshape>) -> tensorshape { return std::vector<size_t>{1}; };
		// scalar shape
		forward = new transfer_func<T>(
		[collector](T* dest, std::vector<const T*> src, shape_io shape)
		{
			assert(1 == src.size());
			if (size_t n_ins = shape.ins_[0].n_elems())
			{
				dest[0] = std::accumulate(src[0] + 1, src[0] + n_ins, *src[0], collector);
			}
		});
	}
	return immutable<T>::get(std::vector<inode<T>*>{a}, shaper, forward,
	[index, bprop](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> fnode = args.front().first;
		varptr<T> bnode = args.front().second;
		varptr<T> barg = bprop(bnode, fnode);
		if (index)
		{
			if (barg.get() == bnode.get())
			{
				barg = identity<T>(bnode);
			}
			barg->set_metadata("grouping", *index);
		}
		return barg;
	}, imm_name);
}

template <typename T>
varptr<T> reduce_max (const varptr<T> a, optional<size_t> dimension)
{
	return compress<T>(a,
	[](const T left, const T right) -> T
	{
		return std::max(left, right);
	}, dimension, "reduce_max");
}

template <typename T>
varptr<T> reduce_sum (const varptr<T> a, optional<size_t> dimension)
{
	return compress<T>(a,
	[](const T left, const T right) -> T
	{
		return left + right;
	}, dimension, "reduce_sum");
}

template <typename T>
inline T mean (const T** data, size_t n)
{
	T accum = 0;
	for (size_t i = 0; i < n; i++)
	{
		accum += *data[i];
	}
	return accum / n;
}

template <typename T>
varptr<T> reduce_mean (const varptr<T> a, optional<size_t> dimension)
{
	varptr<T> denom;
	if (dimension)
	{
		denom = shape_dep<T>::get({ a },
		[dimension](tensorshape& s) -> std::vector<size_t>
		{
			return { s.as_list()[*dimension] };
		}, std::vector<size_t>{1},
		nnutils::formatter() << "axis_" << *dimension << "_size");
	}
	else
	{
		denom = shape_dep<T>::get({ a },
		[](tensorshape& s) -> std::vector<size_t>
		{
			return { s.n_elems() };
		}, std::vector<size_t>{1}, "shape_nelems");
	}
	return reduce_sum<T>(a, dimension) / denom;
}

template <typename T>
varptr<T> arg_compress (const varptr<T> a, REDUCE<T> search,
	optional<size_t> dimension, std::string name)
{
	if (nullptr == a.get()) return nullptr;
	std::string imm_name = (bool) dimension ? nnutils::formatter() << name << "_" << *dimension : name;
	if (inode<T>* parent = unary_parent_search(a.get(), imm_name))
	{
		return parent;
	}
	SHAPER shaper;
	transfer_func<T>* forward;
	if (dimension)
	{
		shaper = [dimension](std::vector<tensorshape> shapes) -> tensorshape
		{
			size_t dim = *dimension;
			tensorshape ts = shapes[0];
			ts.assert_is_fully_defined();
			if (dim >= ts.rank())
			{
				throw std::logic_error(nnutils::formatter()
				<< "attempting to obtain arg index along dimension "
				<< dim << " on a " << ts.rank() << " tensor");
			}
			std::vector<size_t> tv = ts.as_list();
			tv[dim] = 1;
			if (tv.size() > 1)
			{
				if (0 == dim)
					// pop front
				{
					tv = std::vector<size_t>(tv.begin()+1, tv.end());
				}
				else if (tv.size()-1 == dim)
				{
					tv.pop_back();
				}
			}
			return tv;
		};
		forward = new transfer_func<T>(
		[dimension, search](T* dest, std::vector<const T*> src, shape_io shape)
		{
			assert(1 == src.size());
			size_t n_elems = shape.outs_.n_elems();
			std::vector<size_t> coords;
			for (size_t i = 0; i < n_elems; i++)
			{
				coords = shape.outs_.coordinate_from_idx(i);
				if (*dimension == coords.size())
				{
					coords.push_back(0);
				}
				else if (*dimension == 0)
				{
					std::vector<size_t> temp = coords;
					coords = {0};
					coords.insert(coords.end(), temp.begin(), temp.end());
				}
				std::vector<T> search_vec;
				for (size_t j = 0, adim = shape.ins_[0].as_list()[*dimension]; j < adim; j++)
				{
					coords[*dimension] = j;
					search_vec.push_back(src[0][shape.ins_[0].flat_idx(coords)]);
				}
				dest[i] = search(search_vec);
			}
		});
	}
	else
	{
		shaper = [](std::vector<tensorshape> inshapes) -> tensorshape
		{
			return std::vector<size_t>{inshapes[0].rank()};
		};
		// scalar shape
		forward = new transfer_func<T>(
		[search](T* dest, std::vector<const T*> src, shape_io shape)
		{
			assert(1 == src.size());
			size_t n_ins = shape.ins_[0].n_elems();
			std::vector<T> search_vec(src[0], src[0] + n_ins);
			dest[0] = search(search_vec);
		});
	}
	return immutable<T>::get(std::vector<inode<T>*>{a}, shaper, forward,
	[](std::vector<std::pair<inode<T>*,inode<T>*> >)
	{
		// arg_compression's gradient has no intrinsic meaning
		throw std::logic_error("attempting to get gradient of arg compression: undefined and meaningless operation");
		return nullptr;
	}, imm_name);
}

template <typename T>
varptr<T> arg_max (const varptr<T> a, optional<size_t> dimension)
{
	return arg_compress<T>(a,
	[](std::vector<T> data) -> T
	{
		auto mit = std::max_element(data.begin(), data.end(), [](T a, T b)->bool { return a < b; });
		return std::distance(data.begin(), mit);
	}, dimension, "arg_max");
}

template <typename T>
varptr<T> flip (const varptr<T> a, std::vector<size_t> dims)
{
	if (nullptr == a.get()) return nullptr;
	std::unordered_set<inode<T>*> audience;
	std::stringstream ss;
	ss << "flip";
	for (size_t d : dims)
	{
		ss << "_" << d;
	}
	std::string opname = ss.str();
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}

	immutable<T>* sym = immutable<T>::get(std::vector<inode<T>*>{a},
	[](std::vector<tensorshape> shapes) -> tensorshape
	{
		return shapes[0];
	},
	new transfer_func<T>(
	[dims](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::vector<size_t> outlist = shape.outs_.as_list();
		for (size_t i = 0; i < n_elems; i++)
		{
			std::vector<size_t> coord = shape.outs_.coordinate_from_idx(i);
			for (size_t d : dims)
			{
				coord[d] = outlist[d] - coord[d] - 1;
			}
			dest[i] = src[0][shape.ins_[0].flat_idx(coord)];
		}
	}),
	[dims](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		return flip<T>(args.front().second, dims);
	}, opname);
	return sym;
}

template <typename T>
varptr<T> cross_corr2d (const varptr<T> a, const varptr<T> filter,
	std::pair<size_t, size_t> dims)
{
	if (nullptr == a.get() || nullptr == filter.get()) return nullptr;
	std::unordered_set<inode<T>*> audience;
	std::string opname = nnutils::formatter() << "cross_conv_" << dims.first << "_" << dims.second;
	if (inode<T>* parent = ordered_binary_parent_search(a.get(), filter.get(), opname))
	{
		return parent;
	}

	immutable<T>* cv = immutable<T>::get(std::vector<inode<T>*>{a, filter},
	[dims](std::vector<tensorshape> shapes) -> tensorshape
	{
		std::vector<size_t> outshape = shapes[0].as_list();
		std::vector<size_t> filtshape = shapes[1].as_list();
		outshape[dims.first] -= filtshape[dims.first] + 1;
		outshape[dims.second] -= filtshape[dims.second] + 1;
		return outshape;
	},
	new transfer_func<T>(
	[dims](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(2 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::vector<size_t> outlist = shape.outs_.as_list();
		std::vector<size_t> inlist = shape.ins_[0].as_list();
		size_t firstn = inlist[dims.first] - outlist[dims.first];
		size_t secondn = inlist[dims.second] - outlist[dims.second];
		std::vector<size_t> coord;
		for (size_t i = 0; i < n_elems; i++)
		{
			dest[i] = 0;
			coord = shape.outs_.coordinate_from_idx(i);
			for (size_t j = 0; j < firstn; j++)
			{
				for (size_t k = 0; k < secondn; k++)
				{
					dest[i] += src[0][shape.ins_[0].flat_idx(coord)] * src[0][k * firstn + j];

					coord[dims.second]++;
				}
				coord[dims.first]++;
			}
		}
	}),
	[](std::vector<std::pair<inode<T>*,inode<T>*> >)
	{
		throw std::bad_function_call(); // NOT IMPLEMENTED
		return constant<T>::get_shared_one();
	}, opname);

	return cv;
}

template <typename T>
varptr<T> conv2d (const varptr<T> a, const varptr<T> filter,
	std::pair<size_t, size_t> dims)
{
	return cross_corr2d<T>(a, flip<T>(filter, {dims.first, dims.second}), dims);
}

}

#endif
