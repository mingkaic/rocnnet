//
//  elementary.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_ELEMENTARY_HPP

namespace nnet
{

inline tensorshape elementary_shaper (std::vector<tensorshape> shapes)
{
	tensorshape lastshape = shapes[0];
	for (size_t i = 1, nshapes = shapes.size(); i < nshapes; ++i)
	{
		if (shapes[i].n_elems() == 1 || lastshape.n_elems() == 1)
		{
			lastshape = shapes[i];
			continue;
		}
		if (false == shapes[i].is_compatible_with(lastshape))
		{
			std::stringstream ss;
			ss << "shape ";
			print_shape(shapes[i], ss);
			ss << " is incompatible with shape ";
			print_shape(lastshape, ss);
			throw std::runtime_error(ss.str());
		}
		lastshape = shapes[i];
	}
	if (false == lastshape.is_fully_defined()) return std::vector<size_t>{1};
	return lastshape;
}

template <typename T>
inline void elementary_check (const varptr<T>& a, const varptr<T>& b, transfer_func<T>* trans)
{
	if (a->good_status() && b->good_status())
		trans->calc_shape({a->get_shape(), b->get_shape()});
}

template <typename T>
static varptr<T> add_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		if (*a == (T)0)
		{
			return b;
		}
		if (1 == a->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(a);
			return outconst[0] + b;
		}
	}
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	{
		if (*b == (T)0)
		{
			return a;
		}
		if (1 == b->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(b);
			return a + outconst[0];
		}
	}
	elementary_check(a, b, Nf);
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 &&
				((args[0] == a.get() && args[1] == b.get()) ||
				 (args[1] == a.get() && args[0] == b.get())))
				return aud;
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, Nf, ginit, opname);
}

template <typename T>
static varptr<T> sub_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	if (a.get() == b.get()) return constant<T>::get(0);
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		if (*a == (T)0)
		{
			return -b;
		}
		if (1 == a->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(a);
			return outconst[0] - b;
		}
	}
	else if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	{
		if (*b == (T)0)
		{
			return a;
		}
		if (1 == b->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(b);
			return a - outconst[0];
		}
	}
	elementary_check(a, b, Nf);
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a.get() && args[1] == b.get())
				return aud;
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, Nf, ginit, opname);
}

template <typename T>
static varptr<T> mul_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	if (a.get() == b.get()) return pow(a, 2);
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (aconst)
	// optimize only applies to constants
	{
		if (*a == (T)0)
		{
			return constant<T>::get(0);
		}
		if (*a == (T)1)
		{
			return b;
		}
		if (*a == (T)-1)
		{
			return -b;
		}
		if (1 == a->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(a);
			return outconst[0] * b;
		}
	}
	if (bconst)
	// optimize only applies to constants
	{
		if (*b == (T)0)
		{
			return constant<T>::get(0);
		}
		if (*b == (T)1)
		{
			return a;
		}
		if (1 == b->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(b);
			return a * outconst[0];
		}
	}
	elementary_check(a, b, Nf);
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 &&
				((args[0] == a.get() && args[1] == b.get()) ||
				 (args[1] == a.get() && args[0] == b.get())))
				return aud;
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, Nf, ginit, opname);
}

template <typename T>
static varptr<T> div_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	if (a.get() == b.get()) return constant<T>::get(1);
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (aconst)
	{
		// don't allow infinity
		if (*a == (T)0)
		{
			return constant<T>::get(0);
		}
		if (1 == a->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(a);
			return outconst[0] / b;
		}
	}
	if (bconst)
		// optimize only applies to constants
	{
		if (*b == (T)0)
		{
			throw std::logic_error("divide by constant node of value zero");
		}
		if (*b == (T)1)
		{
			return a;
		}
		if (1 == b->get_shape().n_elems())
		{
			std::vector<T> outconst = expose<T>(b);
			return a / outconst[0];
		}
	}
	elementary_check(a, b, Nf);
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a.get() && args[1] == b.get())
				return aud;
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, Nf, ginit, opname);
}

template <typename T>
inline transfer_func<T>* unary_elem_agg (ELEM_FUNC<T> aggregate)
{
	return new transfer_func<T>(elementary_shaper,
	{
		[](size_t i, tensorshape&, const tensorshape&) -> std::vector<size_t> { return {i}; }
	}, aggregate);
}

template <typename T>
inline transfer_func<T>* binary_elem_agg (ELEM_FUNC<T> aggregate)
{
	return new transfer_func<T>(elementary_shaper,
	{
		[](size_t i, tensorshape& ashape, const tensorshape&) -> std::vector<size_t>
		{
			if (1 == ashape.n_elems()) return {0};
			return {i};
		},
		[](size_t i, tensorshape& bshape, const tensorshape&) -> std::vector<size_t>
		{
			if (1 == bshape.n_elems()) return {0};
			return {i};
		}
	}, aggregate);
}

inline OUT_MAPPER get_axis_mapper (size_t axis)
{
	return [axis](size_t i, tensorshape& inshape, const tensorshape& outshape) -> std::vector<size_t>
	{
		if (1 == inshape.n_elems()) return {0};
		std::vector<size_t> olist = outshape.as_list();
		std::vector<size_t> ilist = inshape.as_list();
		std::vector<size_t> coord = outshape.coordinate_from_idx(i);
		if (axis == 0 && ilist[axis] != olist[axis])
		{
			if (ilist[axis] == olist[axis+1])
				coord = std::vector<size_t>(coord.begin()+1, coord.end());
			else
			{
				std::stringstream ss;
				ss << "failed to map ";
				print_shape(outshape, ss);
				ss << " to ";
				print_shape(inshape, ss);
				ss << " along axis 0";
				throw std::logic_error(ss.str());
			}
		}
		else if (axis >= ilist.size() || (ilist[axis] == 1 && olist[axis] > 1))
		{
			coord[axis] = 0;
		}
		i = inshape.sequential_idx(coord);
		return { i };
	};
}

template <typename T>
inline transfer_func<T>* binary_axial_agg (ELEM_FUNC<T> aggregate, size_t axis)
{
	return new transfer_func<T>(
	[axis](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape lastshape = shapes[0];
		for (size_t i = 1, nshapes = shapes.size(); i < nshapes; ++i)
		{
			if (shapes[i].n_elems() == 1 || lastshape.n_elems() == 1)
			{
				lastshape = shapes[i];
				continue;
			}
			std::vector<size_t> prevshape = lastshape.as_list();
			std::vector<size_t> currshape = shapes[i].as_list();
			if (1 == std::abs((double)prevshape.size() - (double)currshape.size()))
			{
				std::vector<size_t>& smallshape = prevshape.size() < currshape.size() ? prevshape : currshape;
				if (axis == 0)
				{
					std::vector<size_t> temp = smallshape;
					smallshape = {1};
					smallshape.insert(smallshape.end(), temp.begin(), temp.end());
				}
				else
				{
					smallshape.push_back(1);
				}
			}
			size_t maxaxal = std::max(prevshape[axis], currshape[axis]);
			prevshape[axis] = currshape[axis] = 1;
			if (false == tensorshape(prevshape).is_compatible_with(tensorshape(currshape)))
			{
				std::stringstream ss;
				ss << "shape ";
				print_shape(shapes[i], ss);
				ss << " is incompatible with shape ";
				print_shape(lastshape, ss);
				ss << " along axis " << axis;
				throw std::runtime_error(ss.str());
			}
			currshape[axis] = maxaxal;
			lastshape = currshape;
		}
		if (false == lastshape.is_fully_defined()) return std::vector<size_t>{1};
		return lastshape;
	},
	{ get_axis_mapper(axis), get_axis_mapper(axis) }, aggregate);
}

template <typename T>
varptr<T> identity (varptr<T> x)
{
	std::string opname = "identity";
	std::unordered_set<inode<T>*> audience;
	if (x->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{x},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> grad;
		args.front()->get_leaf(grad, leaf);
		return grad;
	}, opname);
	out->extract_metadata(x.get());
	return out;
}

template <typename T>
varptr<T> operator + (const varptr<T> a)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::abs(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "abs";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return +(*group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> grad;
		args.front()->get_leaf(grad, leaf);
		return +grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator - (const varptr<T> a)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = -acv;
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	else if (iconnector<T>* aconn = dynamic_cast<iconnector<T>*>(a.get()))
	{
		std::vector<inode<T>*> childargs = aconn->get_arguments();
		if (0 == a->get_label().compare("neg") && 1 == childargs.size())
		{
			// avoids double negatives by calling child directly
			return static_cast<inode<T>*>(childargs[0]);
		}
	}
	std::string opname = "neg";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return -(*group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> grad;
		args.front()->get_leaf(grad, leaf);
		return -grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> sin (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::sin(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "sin";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return std::sin((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sin'(f(x)) = f'(x)*cos(f(x))
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return grad * cos(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> cos (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::cos(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "cos";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return std::cos((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// cos'(f(x)) = -f'(x)*sin(f(x))
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return -grad * sin(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> tan (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::tan(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "tan";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return std::tan((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sec'(f(x)) = f'(x)*sec^2(f(x))
		// better with = f'(x)/cos^2(f(x))
		varptr<T> a = args.front();
		varptr<T> denom = cos(a);
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return grad / (denom * denom);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> csc (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = 1 / std::sin(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "csc";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return 1/std::sin((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
		// better with -f'(x)/(sin(f(x)*tan(f(x))))
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return -grad / (sin(a) * tan(a));
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> sec (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = 1/std::cos(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "sec";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return 1/std::cos((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
		// better with f'(x)*tan(f(x))/cos(f(x))
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return grad * tan(a) / cos(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> cot (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::cos(acv) / std::sin(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "cot";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return std::cos((*group[0])) / std::sin((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// cot'(f(x)) = -f'(x)*csc^2(f(x))
		varptr<T> a = args.front();
		varptr<T> b = csc(a);
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return -grad * b * b;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> exp (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::exp(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "exp";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return std::exp((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return grad * exp(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> sqrt (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::sqrt(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "sqrt";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return std::sqrt((*group[0]));
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sqrt'(f(x)) = f'(x)/(2*sqrt(f(x)))
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return grad / ((T)2 * sqrt(a));
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> pow (const varptr<T> a, double scalar)
{
	if (nullptr == a) return nullptr;
	if (scalar == 0)
	{
		return constant<T>::get(1);
	}
	else if (scalar == 1)
	{
		return a;
	}
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::pow(acv, scalar);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "pow_" << scalar;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([scalar](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return std::pow((*group[0]), scalar);
	})),
	[scalar](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sqrt'(f(x)) = f'(x) * (scalar*f(x)^(scalar-1))
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return (T)scalar * grad * pow(a, scalar-1);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max)
{
	assert(min < max); // todo: maybe throw to indicate usage error
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			if (min > acv) acv = min;
			else if (max < acv) acv = max;
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "clip_val_" << min << "_" << max;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([min, max](const T** group, size_t n) -> T
	{
		assert(n == 1);
		T v = *(group[0]);
		if (min > v) v = min;
		else if (max < v) v = max;
		return v;
	})),
	[min, max](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
		return grad * clip_val(a, min, max);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> l2norm (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		T l2norm = 0;
		std::vector<T> acvec = expose<T>(aconst);
		for (T acv : acvec)
		{
			l2norm += acv * acv;
		}
		return constant<T>::get(l2norm);
	}
	std::string opname = "l2norm";
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	new transfer_func<T>(
	[](std::vector<tensorshape>) { return std::vector<size_t>{1}; },
	{
		[](size_t, tensorshape& ashape, const tensorshape&)
		{
			std::vector<size_t> outidx;
			for (size_t j = 0, n = ashape.n_elems(); j < n; j++)
			{
				outidx.push_back(j);
			}
			return outidx;
		}
	},
	[](const T** group, size_t n)
	{
		T l2norm = 0;
		for (size_t i = 0; i < n; i++)
		{
			l2norm += *(group[i]) * *(group[i]);
		}
		return std::sqrt(l2norm);
	}),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> grad;
		args[0]->get_leaf(grad, leaf);
		return l2norm(grad);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap)
{
	assert(cap > 0); // todo: maybe throw to indicate usage error
	if (nullptr == a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		T l2norm = 0;
		std::vector<T> acvec = expose<T>(aconst);
		for (T acv : acvec)
		{
			l2norm += acv * acv;
		}
		for (T& acv : acvec)
		{
			if (l2norm > cap)
			{
				// normalize
				acv = acv * cap / l2norm;
			}
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "clip_l2norm_" << cap;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{l2norm(a), a},
	binary_elem_agg(ELEM_FUNC<T>([cap](const T** group, size_t n) -> T
	{
		assert(n == 2);
		T l2norm = *(group[0]);
		T elem = *(group[1]);
		if (l2norm > cap)
		{
			// normalize
			elem = elem * cap / l2norm;
		}
		return elem;
	})),
	[cap](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> a = args.front();
		varptr<T> grad;
		a->get_leaf(grad, leaf);
	   	return grad * clip_norm(a, cap);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template<typename T>
varptr<T> operator + (T a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)b) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (a == (T)0) return b;
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	{
		if (*b == (T)0)
		{
			return constant<T>::get(a);
		}
		std::vector<T> bcvec = expose<T>(bconst);
		for (T& bcv : bcvec)
		{
			bcv = a + bcv;
		}
		return constant<T>::get(bcvec, bconst->get_shape());
	}
	std::string opname = nnutils::formatter() << a << "_add";
	std::unordered_set<inode<T>*> audience;
	if (b->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T** group, size_t n) -> T
	 {
	 	assert(n == 1);
		return a + *(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = g'(x)
		varptr<T> bg;
		args.at(0)->get_leaf(bg, leaf);
		return bg;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator + (const varptr<T> a, T b)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		if (*a == (T)0)
		{
			return constant<T>::get(b);
		}
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = acv + b;
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	if (b == (T)0) return a;
	std::string opname = nnutils::formatter() << "add_" << b;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) + b;
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		varptr<T> ag;
		args.at(0)->get_leaf(ag, leaf);
		return ag;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return add(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return add(a, b, *baxis);
	}
	return add_helper<T>(a, b, "add",
	binary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) + *(group[1]);
	})),
   [](std::vector<inode<T>*> args, variable<T>* leaf)
   {
	   // h'(f(x), g(x)) = f'(x) + g'(x)
	   varptr<T> ag;
	   varptr<T> bg;
	   args.at(0)->get_leaf(ag, leaf);
	   args.at(1)->get_leaf(bg, leaf);
	   return ag + bg;
   });
}

template<typename T>
varptr<T> operator - (T a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)b) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (a == (T)0) return -b;
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	{
		if (*b == (T)0)
		{
			return constant<T>::get(a);
		}
		std::vector<T> bcvec = expose<T>(bconst);
		for (T& bcv : bcvec)
		{
			bcv = a - bcv;
		}
		return constant<T>::get(bcvec, bconst->get_shape());
	}
	std::string opname = nnutils::formatter() << a << "_sub";
	std::unordered_set<inode<T>*> audience;
	if (b->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return a - *(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -g'(x)
		varptr<T> bg;
		args.at(0)->get_leaf(bg, leaf);
		return -bg;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator - (const varptr<T> a, T b)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		if (*a == (T)0)
		{
			return constant<T>::get(-b);
		}
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = acv - b;
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	if (b == (T)0) return a;
	std::string opname = nnutils::formatter() << "sub_" << b;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) - b;
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		varptr<T> ag;
		args.at(0)->get_leaf(ag, leaf);
		return ag;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return sub(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return sub(a, b, *baxis);
	}
	return sub_helper<T>(a, b, "sub",
	binary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) - *(group[1]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		varptr<T> ag;
		varptr<T> bg;
		args.at(0)->get_leaf(ag, leaf);
		args.at(1)->get_leaf(bg, leaf);
		return ag - bg;
	});
}

template<typename T>
varptr<T> operator * (T a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)b) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	// optimize only applies to constants
	{
		if (*b == (T)0 || 0 == a)
		{
			return constant<T>::get(0);
		}
		if (*b == (T)1)
		{
			return constant<T>::get(a);
		}
		std::vector<T> bcvec = expose<T>(bconst);
		for (T& bcv : bcvec)
		{
			bcv = a * bcv;
		}
		return constant<T>::get(bcvec, bconst->get_shape());
	}
	if (0 == a) return constant<T>::get(0);
	if (1 == a) return b;
	if (-1 == a) return -b;
	std::string opname = nnutils::formatter() << a << "_mul";
	std::unordered_set<inode<T>*> audience;
	if (b->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return a * *(group[0]);
	})),
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = c*g'(x)
		varptr<T> bg;
		args.at(0)->get_leaf(bg, leaf);
		return a * bg;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator * (const varptr<T> a, T b)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	// optimize only applies to constants
	{
		if (*a == (T)0 || 0 == b)
		{
			return constant<T>::get(0);
		}
		if (*a == (T)1)
		{
			return constant<T>::get(b);
		}
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = acv * b;
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	if (0 == b) return constant<T>::get(0);
	if (1 == b) return a;
	if (-1 == b) return -a;
	std::string opname = nnutils::formatter() << "mul_" << b;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) * b;
	})),
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = c*f'(x)
		varptr<T> ag;
		args.at(0)->get_leaf(ag, leaf);
		return b * ag;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return mul(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return mul(a, b, *baxis);
	}
	return mul_helper<T>(a, b, "mul",
	binary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) * *(group[1]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr<T> ag;
		varptr<T> bg;
		varptr<T> a = args.at(0);
		varptr<T> b = args.at(1);
		a->get_leaf(ag, leaf);
		b->get_leaf(bg, leaf);
		return ag * b + bg * a;
	});
}

template<typename T>
varptr<T> operator / (T a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)b) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (bconst)
	// optimize only applies to constants
	{
		if (*b == (T)0)
		{
			throw std::logic_error("divide by constant node of value zero");
		}
		if (*b == (T)1)
		{
			return constant<T>::get(a);
		}
		std::vector<T> bcvec = expose<T>(bconst);
		for (T& bcv : bcvec)
		{
			bcv = a / bcv;
		}
		return constant<T>::get(bcvec, bconst->get_shape());
	}
	if (a == (T)0)
	{
		return constant<T>::get(0);
	}
	std::string opname = nnutils::formatter() << a << "_div";
	std::unordered_set<inode<T>*> audience;
	if (b->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return a / *(group[0]);
	})),
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr<T> b = args.at(0);
		varptr<T> bg;
		b->get_leaf(bg, leaf);
		return -a * bg / (b * b);
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator / (const varptr<T> a, T b)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	if (aconst)
	{
		if (*a == (T)0)
		{
			return constant<T>::get(0);
		}
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = acv / b;
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	if (b == 0)
	{
		throw std::logic_error("divide by zero");
	}
	if (b == (T)1) return a;
	std::string opname = nnutils::formatter() << "div_" << b;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) / b;
	})),
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)/c
		varptr<T> ag;
		args.at(0)->get_leaf(ag, leaf);
		return ag / b;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return div(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return div(a, b, *baxis);
	}
	return div_helper<T>(a, b, "div",
	binary_elem_agg(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) / *(group[1]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr<T> ag;
		varptr<T> bg;
		varptr<T> a = args.at(0);
		varptr<T> b = args.at(1);
		a->get_leaf(ag, leaf);
		b->get_leaf(bg, leaf);
		return (ag * b - bg * a) / (b * b);
	});
}

template <typename T>
varptr<T> add (const varptr<T> a, const varptr<T> b, size_t axis)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	return add_helper<T>(a, b, nnutils::formatter() << "add_axis_" << axis,
	binary_axial_agg<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) + *(group[1]);
	}), axis),
	[axis](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> ag;
		varptr<T> bg;
		args.at(0)->get_leaf(ag, leaf);
		args.at(1)->get_leaf(bg, leaf);
		return add(ag, bg, axis);
	});
}

template <typename T>
varptr<T> sub (const varptr<T> a, const varptr<T> b, size_t axis)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	return sub_helper<T>(a, b, nnutils::formatter() << "sub_axis_" << axis,
	binary_axial_agg<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) - *(group[1]);
	}), axis),
	[axis](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> ag;
		varptr<T> bg;
		args.at(0)->get_leaf(ag, leaf);
		args.at(1)->get_leaf(bg, leaf);
		return sub(ag, bg, axis);
	});
}

template <typename T>
varptr<T> mul (const varptr<T> a, const varptr<T> b, size_t axis)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	return mul_helper<T>(a, b, nnutils::formatter() << "mul_axis_" << axis,
	binary_axial_agg<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) * *(group[1]);
	}), axis),
	[axis](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr<T> ag;
		varptr<T> bg;
		varptr<T> a = args.at(0);
		varptr<T> b = args.at(1);
		a->get_leaf(ag, leaf);
		b->get_leaf(bg, leaf);
		return mul(ag, b, axis) + mul(bg, a, axis);
	});
}

template <typename T>
varptr<T> div (const varptr<T> a, const varptr<T> b, size_t axis)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	return div_helper<T>(a, b, nnutils::formatter() << "div_axis_" << axis,
	binary_axial_agg<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) / *(group[1]);
	}), axis),
	[axis](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr<T> ag;
		varptr<T> bg;
		varptr<T> a = args.at(0);
		varptr<T> b = args.at(1);
		a->get_leaf(ag, leaf);
		b->get_leaf(bg, leaf);
		return div(mul(ag, b, axis) - mul(bg, a, axis), pow(b, 2), axis);
	});
}

}

#endif
