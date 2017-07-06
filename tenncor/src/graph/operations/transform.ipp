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
inline T copyover (const T** data, size_t)
{
	if (data[0]) return *data[0]; // 1 to 1 copy over
	return (T)0;
}

template <typename T>
inline transfer_func<T>* unary_trans_agg (OUT_MAPPER indexer, SHAPER shaper)
{
	return new transfer_func<T>(shaper, {indexer}, copyover<T>);
}

template <typename T>
varptr<T> transpose (const varptr<T> a, std::pair<size_t,size_t> axis_swap)
{
	if (nullptr == a) return nullptr;
	assert(2 >= a->get_shape().rank());
	std::string opname = nnutils::formatter() << "transpose_" << axis_swap.first << "_" << axis_swap.second;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_trans_agg<T>(
	[axis_swap](size_t i, tensorshape& ashape, const tensorshape& outshape) -> std::vector<size_t>
	{
		std::vector<size_t> coords = outshape.coordinate_from_idx(i);
		size_t max_axis = std::max(axis_swap.first, axis_swap.second);
		if (max_axis >= coords.size())
		{
			coords.insert(coords.end(), max_axis - coords.size() + 1, 0);
		}
		std::swap(coords[axis_swap.first], coords[axis_swap.second]);
		return { ashape.sequential_idx(coords) };
	},
	[axis_swap](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape ts = shapes[0];
		if (ts.is_fully_defined())
		{
			std::vector<size_t> inl = ts.as_list();
			size_t max_axis = std::max(axis_swap.first, axis_swap.second);
			if (max_axis >= inl.size())
			{
				inl.insert(inl.end(), max_axis - inl.size() + 1, 1);
			}
			std::swap(inl[axis_swap.first], inl[axis_swap.second]);
			return inl;
		}
		return tensorshape();
	}),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> grad;
		args.front()->get_leaf(grad, leaf);
		return transpose(grad);
	}, opname);
}

template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch)
{
	if (nullptr == a || nullptr == watch) return nullptr;
	if (a->good_status() && *a == (T)0) return a;
	// additional constraint that watch shape must be have shape with
	// dimensions greater or equal to a's dimensional value (shape.as_list()[i])
	std::string opname = "fit";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a.get() && args[1] == watch.get())
				return aud;
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, watch},
	new transfer_func<T>(
	[](std::vector<tensorshape> shapes) -> tensorshape
	{
		return shapes[1]; // watch is always argument 2
	},
	{
		[](size_t i, tensorshape& ashape, const tensorshape& outshape) -> std::vector<size_t>
		{
			std::vector<size_t> alist = ashape.as_list();
			std::vector<size_t> coords = outshape.coordinate_from_idx(i);
			bool inbound = true;
			size_t minrank = std::min(alist.size(), coords.size());
			for (size_t j = 0; inbound && j < minrank; j++)
			{
				inbound = coords[j] < alist[j];
			}
			for (size_t j = minrank, rank = coords.size(); inbound && j < rank; j++)
			{
				inbound = coords[j] == 0;
			}
			if (false == inbound)
			{
				return { ashape.n_elems() };
			}
			return { ashape.sequential_idx(coords) };
		},
		[](size_t, tensorshape& ashape, const tensorshape&) -> std::vector<size_t> { return { ashape.n_elems() }; }
	}, copyover<T>),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> grad;
		args.front()->get_leaf(grad, leaf);
		return grad;
	}, opname, watch);
}

template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier)
{
	if (nullptr == a) return nullptr;
	if (multiplier == 0) return constant<T>::get(0);
	if (multiplier == 1) return a;
	std::string opname = nnutils::formatter() << "extend_" << index << "_" << multiplier;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_trans_agg<T>(
	[index, multiplier](size_t i, tensorshape& ashape, const tensorshape& outshape) -> std::vector<size_t>
	{
		size_t adim = ashape.as_list()[index];
		std::vector<size_t> coords = outshape.coordinate_from_idx(i);
		coords[index] = coords[index] % adim;
		return { ashape.sequential_idx(coords) };
	},
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
	}),
	[index, multiplier](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> grad;
		args.front()->get_leaf(grad, leaf);
		return grad;
	}, opname);
}

template <typename T>
varptr<T> compress (const varptr<T> a, optional<size_t> index,
	ELEM_FUNC<T> collector, std::string name,
	std::function<varptr<T>(varptr<T>,varptr<T>)> bprop)
{
	if (nullptr == a) return nullptr;
	std::string imm_name = (bool) index ? nnutils::formatter() << name << "_" << *index : name;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(imm_name, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	transfer_func<T>* forward;
	if ((bool) index)
	{
		forward = new transfer_func<T>(
		[index](std::vector<tensorshape> shapes) -> tensorshape
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
		},
		{
			[index](size_t i, tensorshape& ashape, const tensorshape& outshape) -> std::vector<size_t>
			{
				if (ashape.rank() <= *index) return {i};
				std::vector<size_t> outidx;
				std::vector<size_t> coords = outshape.coordinate_from_idx(i);
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
				for (size_t j = 0, adim = ashape.as_list()[*index]; j < adim; j++)
				{
					coords[*index] = j;
					outidx.push_back(ashape.sequential_idx(coords));
				}
				return outidx;
			}
		}, collector);
	}
	else
	{
		// scalar shape
		forward = new transfer_func<T>(
		[](std::vector<tensorshape>) -> tensorshape { return std::vector<size_t>{1}; },
		{
			[](size_t, tensorshape &ashape, const tensorshape&)
			{
				std::vector<size_t> outidx;
				for (size_t j = 0, n = ashape.n_elems(); j < n; j++)
				{
					outidx.push_back(j);
				}
				return outidx;
			}
		}, collector);
	}
	return immutable<T>::get(std::vector<inode<T>*>{a}, forward,
	[index, bprop](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> bnode;
		varptr<T> fnode = args.front();
		fnode->get_leaf(bnode, leaf);
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
	return compress<T>(a, dimension,
	[](const T** data, size_t n) -> T
	{
		return **std::max_element(data, data+n, [](const T* a, const T* b)->bool { return *a < *b; });
	}, "reduce_max");
}

template <typename T>
varptr<T> reduce_sum (const varptr<T> a, optional<size_t> dimension)
{
	return compress<T>(a, dimension,
	[](const T** data, size_t n) -> T
	{
		T accum = 0;
		for (size_t i = 0; i < n; i++)
		{
			accum += *data[i];
		}
		return accum;
	}, "reduce_sum");
}

template <typename T>
varptr<T> reduce_mean (const varptr<T> a, optional<size_t> dimension)
{
	return compress<T>(a, dimension,
	[](const T** data, size_t n) -> T
	{
		T accum = 0;
		for (size_t i = 0; i < n; i++)
		{
			accum += *data[i];
		}
		return accum / n;
	}, "reduce_mean",
	[dimension](varptr<T> back, varptr<T> forward)
	{
		varptr<T> denom;
		if (dimension)
		{
			denom = shape_dep<T>::get({ forward.get() },
			[dimension](tensorshape& s) -> std::vector<size_t>
			{
				return { s.as_list()[*dimension] };
			}, std::vector<size_t>{1},
			nnutils::formatter() << "axis_" << *dimension << "_size");
		}
		else
		{
			denom = shape_dep<T>::get({ forward.get() },
			[](tensorshape& s) -> std::vector<size_t>
			{
				return { s.n_elems() };
			}, std::vector<size_t>{1}, "shape_nelems");
		}
		return back / denom;
	});
}

template <typename T>
varptr<T> arg_compress (const varptr<T> a, optional<size_t> dimension,
	ELEM_FUNC<T> search, std::string name)
{
	if (nullptr == a) return nullptr;
	std::string imm_name = (bool) dimension ? nnutils::formatter() << name << "_" << *dimension : name;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(imm_name, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	transfer_func<T>* forward;
	if (dimension)
	{
		forward = new transfer_func<T>(
		[dimension](std::vector<tensorshape> shapes) -> tensorshape
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
		},
		{
			[dimension](size_t i, tensorshape& ashape, const tensorshape& outshape)
			{
				std::vector<size_t> outidx;
				std::vector<size_t> coords = outshape.coordinate_from_idx(i);
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
				for (size_t j = 0, adim = ashape.as_list()[*dimension]; j < adim; j++)
				{
					coords[*dimension] = j;
					outidx.push_back(ashape.sequential_idx(coords));
				}
				return outidx;
			}
		}, search);
	}
	else
	{
		// scalar shape
		forward = new transfer_func<T>(
		[](std::vector<tensorshape> inshapes) -> tensorshape
		{
			return std::vector<size_t>{inshapes[0].rank()};
		},
		{
			[](size_t, tensorshape &ashape, const tensorshape&)
			{
				std::vector<size_t> outidx;
				for (size_t j = 0, n = ashape.n_elems(); j < n; j++)
				{
					outidx.push_back(j);
				}
				return outidx;
			}
		}, search);
	}
	return immutable<T>::get(std::vector<inode<T>*>{a}, forward,
	[dimension, search](std::vector<inode<T>*>, variable<T>*)
	{
		// arg_compression's gradient has no intrinsic meaning
		throw std::logic_error("attempting to get gradient of arg compression: undefined and meaningless operation");
		return nullptr;
	}, imm_name);
}

template <typename T>
varptr<T> arg_max (const varptr<T> a, optional<size_t> dimension)
{
	return arg_compress<T>(a, dimension,
	[](const T** data, size_t n) -> T
	{
		auto mit = std::max_element(data, data+n, [](const T* a, const T* b)->bool { return *a < *b; });
		return std::distance(data, mit);
	}, "arg_max");
}

}

#endif
