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
	tensorshape lastshape;
	for (size_t i = 0, nshapes = shapes.size(); i < nshapes; ++i)
	{
		if (shapes[i].n_elems() == 1)
		{
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
static varptr<T> add_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	varptr<T> out = nullptr;
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (aconst && *aconst == (T) 0)
	{
		out = b;
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		out = outconst[0] + b;
	}
	else if (bconst && *bconst == (T) 0)
	{
		out = a;
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(bconst);
		out = a + outconst[0];
	}
	if (nullptr != out.get())
	{
		delete Nf;
		return out;
	}
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
			{
				delete Nf;
				return aud;
			}
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, Nf, ginit, opname);
}

template <typename T>
static varptr<T> sub_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	varptr<T> out = nullptr;
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (a.get() == b.get())
	{
		out = constant<T>::get(0);
	}
	else if (aconst && *aconst == (T) 0)
	{
		out = -b;
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		out = outconst[0] - b;
	}
	else if (bconst && *bconst == (T) 0)
	{
		out = a;
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(bconst);
		out = a - outconst[0];
	}
	if (nullptr != out.get())
	{
		delete Nf;
		return out;
	}
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a.get() && args[1] == b.get())
			{
				delete Nf;
				return aud;
			}
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, Nf, ginit, opname);
}

template <typename T>
static varptr<T> mul_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	varptr<T> out = nullptr;
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (a.get() == b.get())
	{
		out = pow(a, 2);
	}
	else if (aconst && *aconst == (T) 0)
	{
		out = constant<T>::get(0);
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		out = outconst[0] * b;
	}
	else if (bconst && *bconst == (T) 0)
	{
		out = constant<T>::get(0);
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(b);
		out = a * outconst[0];
	}
	if (nullptr != out.get())
	{
		delete Nf;
		return out;
	}
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
			{
				delete Nf;
				return aud;
			}
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, Nf, ginit, opname);
}

template <typename T>
static varptr<T> div_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, transfer_func<T>* Nf, BACK_MAP<T> ginit)
{
	varptr<T> out = nullptr;
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (a.get() == b.get())
	{
		out = constant<T>::get(1);
	}
	else if (aconst && *aconst == (T)0)
	{
		out = constant<T>::get(0);
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		out = outconst[0] / b;
	}
	else if (bconst && *bconst == (T)0)
	// don't allow infinity
	{
		delete Nf;
		throw std::logic_error("divide by constant node of value zero");
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(bconst);
		out = a / outconst[0];
	}
	if (nullptr != out.get())
	{
		delete Nf;
		return out;
	}
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a.get() && args[1] == b.get())
			{
				delete Nf;
				return aud;
			}
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

inline std::vector<size_t> injective (size_t i, tensorshape& ashape, const tensorshape&)
{
	if (1 == ashape.n_elems()) return {0};
	return {i};
}

template <typename T>
inline transfer_func<T>* binary_elem_agg (ELEM_FUNC<T> aggregate)
{
	return new transfer_func<T>(elementary_shaper, { injective, injective }, aggregate);
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
inline transfer_func<T>* binary_axial_left (ELEM_FUNC<T> aggregate, size_t axis)
{
	return new transfer_func<T>(
	[axis](std::vector<tensorshape> shapes) -> tensorshape
			{
		tensorshape shape1 = shapes[0];
		tensorshape shape2 = shapes[1];

		if (shape2.n_elems() == 1) return shape1;

		std::vector<size_t> s2list = shape2.as_list();

		if (axis == 0)
		{
			s2list = std::vector<size_t>(s2list.begin()+1, s2list.end());
		}
		else if (axis < s2list.size())
		{
			s2list[axis] = 1;
		}
		else
		{
			s2list.push_back(1);
		}
		if (false == shape1.is_compatible_with(tensorshape(s2list)))
		{
			std::stringstream ss;
			ss << "shape ";
			print_shape(shape1, ss);
			ss << " is incompatible with shape ";
			print_shape(shape2, ss);
			ss << " along axis " << axis;
			throw std::runtime_error(ss.str());
		}

		if (false == shape2.is_fully_defined()) return std::vector<size_t>{1};
		return shape2;
	},
	{ get_axis_mapper(axis), injective }, aggregate);
}

template <typename T>
inline transfer_func<T>* binary_axial_right (ELEM_FUNC<T> aggregate, size_t axis)
{
	return new transfer_func<T>(
	[axis](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape shape1 = shapes[0];
		tensorshape shape2 = shapes[1];

		if (shape1.n_elems() == 1) return shape2;

		std::vector<size_t> s1list = shape1.as_list();

		if (axis == 0)
		{
			s1list = std::vector<size_t>(s1list.begin()+1, s1list.end());
		}
		else if (axis < s1list.size())
		{
			s1list[axis] = 1;
		}
		else
		{
			s1list.push_back(1);
		}
		if (false == shape2.is_compatible_with(tensorshape(s1list)))
		{
			std::stringstream ss;
			ss << "shape ";
			print_shape(shape1, ss);
			ss << " is incompatible with shape ";
			print_shape(shape2, ss);
			ss << " along axis " << axis;
			throw std::runtime_error(ss.str());
		}

 		if (false == shape1.is_fully_defined()) return std::vector<size_t>{1};
		return shape1;
	},
	{ injective, get_axis_mapper(axis) }, aggregate);
}

template <typename T>
inline void axial_set_jacobian (varptr<T>& root, const varptr<T>& branch, size_t axis)
{
	if (iconnector<T>* iconn = dynamic_cast<iconnector<T>*>(root.get()))
	{
		std::unordered_set<ileaf<T>*> temp = branch->get_leaves();
		std::vector<variable<T>*> leef;
		for (ileaf<T>* ilef : temp)
		{
			if (variable<T>* var = dynamic_cast<variable<T>*>(ilef))
			{
				leef.push_back(var);
			}
		}
		iconn->set_jacobian([axis](inode<T>* root, NODE_MAN<T>) -> inode<T>*
		{
			return reduce_sum(varptr<T>(root), axis);
		}, leef);
	}
}

template <typename T>
varptr<T> identity (varptr<T> x)
{
	if (nullptr == x.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> grad = args.front().second;
		return grad;
	}, opname);
	out->extract_metadata(x.get());
	return out;
}

template <typename T>
varptr<T> operator + (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> grad = args.front().second;
		return +grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator - (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> grad = args.front().second;
		return -grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> sin (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// sin'(f(x)) = f'(x)*cos(f(x))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return grad * cos(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> cos (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// cos'(f(x)) = -f'(x)*sin(f(x))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return -grad * sin(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> tan (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// sec'(f(x)) = f'(x)*sec^2(f(x))
		// better with = f'(x)/cos^2(f(x))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		varptr<T> denom = cos(a);
		return grad / (denom * denom);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> csc (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
		// better with -f'(x)/(sin(f(x)*tan(f(x))))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return -grad / (sin(a) * tan(a));
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> sec (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
		// better with f'(x)*tan(f(x))/cos(f(x))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return grad * tan(a) / cos(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> cot (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// cot'(f(x)) = -f'(x)*csc^2(f(x))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		varptr<T> b = csc(a);
		return -grad * b * b;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> exp (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return grad * exp(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> sqrt (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// sqrt'(f(x)) = f'(x)/(2*sqrt(f(x)))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return grad / ((T)2 * sqrt(a));
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> pow (const varptr<T> a, double scalar)
{
	if (nullptr == a.get()) return nullptr;
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
	[scalar](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// sqrt'(f(x)) = f'(x) * (scalar*f(x)^(scalar-1))
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return (T)scalar * grad * pow(a, scalar-1);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max)
{
	assert(min < max); // todo: maybe throw to indicate usage error
	if (nullptr == a.get()) return nullptr;
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
	[min, max](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
		return grad * clip_val(a, min, max);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> l2norm (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
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
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> grad = args.front().second;
		return l2norm(grad);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap)
{
	assert(cap > 0); // todo: maybe throw to indicate usage error
	if (nullptr == a.get()) return nullptr;
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
	[cap](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
	   	return grad * clip_norm(a, cap);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template<typename T>
varptr<T> operator + (T a, const varptr<T> b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (a == (T)0) return b;
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	{
		if (*bconst == (T)0)
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
	unary_elem_agg(ELEM_FUNC<T>(
	[a](const T** group, size_t n) -> T
	 {
	 	assert(n == 1);
		return a + *(group[0]);
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(c, g(x)) = g'(x)
		varptr<T> grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator + (const varptr<T> a, T b)
{
	if (nullptr == a.get()) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		if (*aconst == (T)0)
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
	unary_elem_agg(ELEM_FUNC<T>(
	[b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) + b;
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), c) = f'(x)
		varptr<T> grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return add_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return add_axial_a(a, b, *baxis);
	}
	return add_helper<T>(a, b, "add",
	binary_elem_agg(ELEM_FUNC<T>(
	[](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) + *(group[1]);
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return ag + bg;
	});
}

template<typename T>
varptr<T> operator - (T a, const varptr<T> b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (a == (T)0) return -b;
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	{
		if (*bconst == (T)0)
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
	unary_elem_agg(ELEM_FUNC<T>(
	[a](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return a - *(group[0]);
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(c, g(x)) = -g'(x)
		varptr<T> grad = args.at(0).second;
		return -grad;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator - (const varptr<T> a, T b)
{
	if (nullptr == a.get()) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		if (*aconst == (T)0)
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
	unary_elem_agg(ELEM_FUNC<T>(
	[b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) - b;
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), c) = f'(x)
		varptr<T> grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return sub_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return sub_axial_a(a, b, *baxis);
	}
	return sub_helper<T>(a, b, "sub",
	binary_elem_agg(ELEM_FUNC<T>(
	[](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) - *(group[1]);
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return ag - bg;
	});
}

template<typename T>
varptr<T> operator * (T a, const varptr<T> b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	// optimize only applies to constants
	{
		if (*bconst == (T)0 || 0 == a)
		{
			return constant<T>::get(0);
		}
		if (*bconst == (T)1)
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
	if ((T) 0 == a) return constant<T>::get(0);
	if ((T) 1 == a) return b;
	if ((T) -1 == a) return -b;
	std::string opname = nnutils::formatter() << a << "_mul";
	std::unordered_set<inode<T>*> audience;
	if (b->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>(
	[a](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return a * *(group[0]);
	})),
	[a](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(c, g(x)) = c*g'(x)
		varptr<T> grad = args.at(0).second;
		return a * grad;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator * (const varptr<T> a, T b)
{
	if (nullptr == a.get()) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	// optimize only applies to constants
	{
		if (*aconst == (T)0 || 0 == b)
		{
			return constant<T>::get(0);
		}
		if (*aconst == (T)1)
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
	if ((T) 0 == b) return constant<T>::get(0);
	if ((T) 1 == b) return a;
	if ((T) -1 == b) return -a;
	std::string opname = nnutils::formatter() << "mul_" << b;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>(
	[b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) * b;
	})),
	[b](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), c) = c*f'(x)
		varptr<T> grad = args.at(0).second;
		return b * grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return mul_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return mul_axial_a(a, b, *baxis);
	}
	return mul_helper<T>(a, b, "mul",
	binary_elem_agg(ELEM_FUNC<T>(
	[](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) * *(group[1]);
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr<T> a = args.at(0).first;
		varptr<T> b = args.at(1).first;
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return ag * b + bg * a;
	});
}

template<typename T>
varptr<T> operator / (T a, const varptr<T> b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (bconst)
	// optimize only applies to constants
	{
		if (*bconst == (T)0)
		{
			throw std::logic_error("divide by constant node of value zero");
		}
		if (*bconst == (T)1)
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
	unary_elem_agg(ELEM_FUNC<T>(
	[a](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return a / *(group[0]);
	})),
	[a](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr<T> b = args.at(0).first;
		varptr<T> bg = args.at(0).second;
		return -a * bg / (b * b);
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template<typename T>
varptr<T> operator / (const varptr<T> a, T b)
{
	if (nullptr == a.get()) return nullptr;
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	if (aconst)
	{
		if (*aconst == (T)0)
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
	if (b == (T) 0)
	{
		throw std::logic_error("divide by zero");
	}
	if (b == (T) 1) return a;
	std::string opname = nnutils::formatter() << "div_" << b;
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>(
	[b](const T** group, size_t n) -> T
	{
		assert(n == 1);
		return *(group[0]) / b;
	})),
	[b](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), c) = f'(x)/c
		varptr<T> ag = args.at(0).second;
		return ag / b;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return div_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return div_axial_a(a, b, *baxis);
	}
	return div_helper<T>(a, b, "div",
	binary_elem_agg(ELEM_FUNC<T>(
	[](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) / *(group[1]);
	})),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr<T> a = args.at(0).first;
		varptr<T> b = args.at(1).first;
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return (ag * b - bg * a) / (b * b);
	});
}

template <typename T>
varptr<T> add_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> addout = add_helper<T>(a, b, nnutils::formatter() << "add_axis_a_" << axis_a,
	binary_axial_left(ELEM_FUNC<T>(
	[](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) + *(group[1]);
	}), axis_a),
	[axis_a](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return add_axial_a(ag, bg, axis_a);
	});
	axial_set_jacobian(addout, a, axis_a);
	return addout;
}

template <typename T>
varptr<T> add_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> addout = add_helper<T>(a, b, nnutils::formatter() << "add_axis_b_" << axis_b,
	binary_axial_right(
	ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) + *(group[1]);
	}), axis_b),
	[axis_b](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return add_axial_b(ag, bg, axis_b);
	});
	axial_set_jacobian(addout, b, axis_b);
	return addout;
}

template <typename T>
varptr<T> sub_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> subout = sub_helper<T>(a, b, nnutils::formatter() << "sub_axis_a_" << axis_a,
	binary_axial_left<T>(ELEM_FUNC<T>(
	[](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) - *(group[1]);
	}), axis_a),
	[axis_a](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return sub_axial_a(ag, bg, axis_a);
	});
	axial_set_jacobian(subout, a, axis_a);
	return subout;
}

template <typename T>
varptr<T> sub_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> subout = sub_helper<T>(a, b, nnutils::formatter() << "sub_axis_b_" << axis_b,
	binary_axial_right<T>(ELEM_FUNC<T>(
	[](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) - *(group[1]);
	}), axis_b),
	[axis_b](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return sub_axial_b(ag, bg, axis_b);
	});
	axial_set_jacobian(subout, b, axis_b);
	return subout;
}

template <typename T>
varptr<T> mul_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> mulout = mul_helper<T>(a, b, nnutils::formatter() << "mul_axis_a_" << axis_a,
	binary_axial_left<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) * *(group[1]);
	}), axis_a),
	[axis_a](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr<T> a = args.at(0).first;
		varptr<T> b = args.at(1).first;
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return mul_axial_a(ag, b, axis_a) + mul_axial_a(a, bg, axis_a);
	});
	axial_set_jacobian(mulout, a, axis_a);
	return mulout;
}

template <typename T>
varptr<T> mul_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> mulout = mul_helper<T>(a, b, nnutils::formatter() << "mul_axis_b_" << axis_b,
	binary_axial_right<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) * *(group[1]);
	}), axis_b),
	[axis_b](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr<T> a = args.at(0).first;
		varptr<T> b = args.at(1).first;
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return mul_axial_b(ag, b, axis_b) + mul_axial_b(a, bg, axis_b);
	});
	axial_set_jacobian(mulout, b, axis_b);
	return mulout;
}

template <typename T>
varptr<T> div_axial_a (const varptr<T> a, const varptr<T> b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> divout = div_helper<T>(a, b, nnutils::formatter() << "div_axis_a_" << axis_a,
	binary_axial_left<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) / *(group[1]);
	}), axis_a),
	[axis_a](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr<T> a = args.at(0).first;
		varptr<T> b = args.at(1).first;
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return (mul_axial_a(ag, b, axis_a) - mul_axial_b(bg, a, axis_a)) / pow(b, 2);
	});
	axial_set_jacobian(divout, a, axis_a);
	return divout;
}

template <typename T>
varptr<T> div_axial_b (const varptr<T> a, const varptr<T> b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	varptr<T> divout = div_helper<T>(a, b, nnutils::formatter() << "div_axis_b_" << axis_b,
	binary_axial_right<T>(ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		assert(n == 2);
		return *(group[0]) / *(group[1]);
	}), axis_b),
	[axis_b](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr<T> a = args.at(0).first;
		varptr<T> b = args.at(1).first;
		varptr<T> ag = args.at(0).second;
		varptr<T> bg = args.at(1).second;
		return (mul_axial_b(ag, b, axis_b) - mul_axial_a(bg, a, axis_b)) / pow(b, 2);
	});
	axial_set_jacobian(divout, b, axis_b);
	return divout;
}

}

#endif
