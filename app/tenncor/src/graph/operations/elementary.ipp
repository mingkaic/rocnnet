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

static inline tensorshape elementary_shaper (std::vector<tensorshape> shapes)
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

static inline SHAPER binary_axial_shape (size_t axis, bool left)
{
	return
	[axis, left](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape shape1;
		tensorshape shape2;
		if (left)
		{
			shape1 = shapes[0];
			shape2 = shapes[1];
		}
		else
		{
			shape1 = shapes[1];
			shape2 = shapes[0];
		}

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
	};
}

template <typename T>
static TRANSFER_FUNC<T> binary_elem_agg (AGGREGATE<T> aggregate)
{
	return [aggregate](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(2 == srcs.size());
		size_t n_out = shapes.outs_.n_elems();
		size_t n_left = shapes.ins_[0].n_elems();
		size_t n_right = shapes.ins_[1].n_elems();
		bool left_mul = n_left > 1;
		bool right_mul = n_right > 1;
		for (size_t i = 0; i < n_out; i++)
		{
			dest[i] = aggregate(srcs[0][i * left_mul], srcs[1][i * right_mul]);
		}
	};
}

template <typename T>
static TRANSFER_FUNC<T> binary_axial (AGGREGATE<T> aggregate, size_t axis, bool left)
{
	short idx = 0;
	if (!left)
	{
		aggregate = [aggregate](T left_arg, T right_arg)
		{
			return aggregate(right_arg, left_arg);
		};
		idx = 1;
	}
	return [aggregate, axis, idx](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(2 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		size_t n_axis = shapes.ins_[idx].n_elems();
		std::vector<size_t> olist = shapes.outs_.as_list();
		std::vector<size_t> ilist = shapes.ins_[idx].as_list();

		std::vector<size_t> coords;
		for (size_t i = 0; i < n_elems; i++)
		{
			size_t axis_idx;
			if (1 < n_axis)
			{
				axis_idx = 0;
			}
			else
			{
				coords = shapes.outs_.coordinate_from_idx(i);
				if (axis == 0 && ilist[axis] != olist[axis])
				{
					if (ilist[axis] == olist[axis+1])
						coords = std::vector<size_t>(coords.begin()+1, coords.end());
					else
					{
						std::stringstream ss;
						ss << "failed to map ";
						print_shape(shapes.outs_, ss);
						ss << " to ";
						print_shape(shapes.ins_[idx], ss);
						ss << " along axis 0";
						throw std::logic_error(ss.str());
					}
				}
				else
				{
					coords[axis] = 0;
				}
				axis_idx = shapes.ins_[idx].flat_idx(coords);
			}
			dest[i] = aggregate(srcs[idx][axis_idx], srcs[(idx + 1) % 2][i]);
		}
	};
}

template <typename T>
static varptr<T> add_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<T> Nf, BACK_MAP<T> ginit)
{
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (aconst && *aconst == (T) 0)
	{
		return b;
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		return outconst[0] + b;
	}
	else if (bconst && *bconst == (T) 0)
	{
		return a;
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(bconst);
		return a + outconst[0];
	}

	if (inode<T>* parent = unordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, shaper, new transfer_func<T>(Nf), ginit, opname);
}

template <typename T>
static varptr<T> sub_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<T> Nf, BACK_MAP<T> ginit)
{
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (a.get() == b.get())
	{
		return constant<T>::get(0);
	}
	else if (aconst && *aconst == (T) 0)
	{
		return -b;
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		return outconst[0] - b;
	}
	else if (bconst && *bconst == (T) 0)
	{
		return a;
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(bconst);
		return a - outconst[0];
	}

	if (inode<T>* parent = ordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, shaper, new transfer_func<T>(Nf), ginit, opname);
}

template <typename T>
static varptr<T> mul_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<T> Nf, BACK_MAP<T> ginit)
{
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (a.get() == b.get())
	{
		return pow(a, 2);
	}
	else if (aconst && *aconst == (T) 0)
	{
		return constant<T>::get(0);
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		return outconst[0] * b;
	}
	else if (bconst && *bconst == (T) 0)
	{
		return constant<T>::get(0);
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(b);
		return a * outconst[0];
	}

	if (inode<T>* parent = unordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, shaper, new transfer_func<T>(Nf), ginit, opname);
}

template <typename T>
static varptr<T> div_helper (const varptr<T>& a, const varptr<T>& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<T> Nf, BACK_MAP<T> ginit)
{
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (a.get() == b.get())
	{
		return constant<T>::get(1);
	}
	else if (aconst && *aconst == (T)0)
	{
		return constant<T>::get(0);
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		return outconst[0] / b;
	}
	else if (bconst && *bconst == (T)0)
	// don't allow infinity
	{
		throw std::logic_error("divide by constant node of value zero");
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(bconst);
		return a / outconst[0];
	}

	if (inode<T>* parent = ordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, shaper, new transfer_func<T>(Nf), ginit, opname);
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
		iconn->set_jacobian_front(
		[axis](inode<T>* root, std::vector<inode<T>*>, std::vector<inode<T>*>) -> inode<T>*
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
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{x}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::memcpy(dest, src[0], sizeof(T) * n_elems);
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return +data; });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return -data; });
	}),
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
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::sin(data); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::cos(data); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::tan(data); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return 1 / std::sin(data); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return 1 / std::cos(data); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::cos(data) / std::sin(data); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::exp(data); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::sqrt(data); });
	}),
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
varptr<T> round (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::round(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "round";
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::round(data); });
	}),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// round'(f(x)) = round(f'(x))
		varptr<T> grad = args.front().second;
		return nnet::round(grad);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> log (const varptr<T> a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = std::log(acv);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = "log";
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const T data) { return std::log(data); });
	}),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// log'(f(x)) = 1 / f(x)
		varptr<T> a = args.front().first;
		return (T) 1 / a;
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([scalar](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[scalar](const T data) { return std::pow(data, scalar); });
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>([min, max](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[min, max](const T data)
		{
			if (min > data) return min;
			else if (max < data) return max;
			return data;
		});
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	[](std::vector<tensorshape>) { return std::vector<size_t>{1}; },
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		// l2norm = sqrt(sum_i=0:n(sqr(xi)))
		dest[0] = std::sqrt(std::accumulate(src[0], src[0] + n_elems, (T) 0,
		[](const T left, const T right)
		{
			return left + right * right;
		}));
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{l2norm(a), a},
	elementary_shaper,
	new transfer_func<T>(
	[cap](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(2 == srcs.size());
		T l2norm = srcs[0][0];
		size_t n_out = shapes.outs_.n_elems();
		std::transform(srcs[1], srcs[1] + n_out, dest,
		[cap, l2norm](const T data)
		{
			return data * cap / l2norm;
		});
	}),
	[cap](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		varptr<T> a = args.front().first;
		varptr<T> grad = args.front().second;
	   	return grad * clip_norm(a, cap);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> conditional (T a, const varptr<T> b, std::function<bool(T,T)> compare, std::string name)
{
	if (nullptr == b.get()) return nullptr;
	if (constant<T>* bconst = dynamic_cast<constant<T>*>(b.get()))
	{
		std::vector<T> bcvec = expose<T>(bconst);
		for (T& bcv : bcvec)
		{
			bcv = (T) compare(a, bcv);
		}
		return constant<T>::get(bcvec, bconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "conditional_" << name << "_" << a;
	std::unordered_set<inode<T>*> audience;
	if (b->find_audience(opname, audience))
	{
		return *audience.begin(); // share nodes when possible
	}

	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	new transfer_func<T>(
	[compare, a](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		size_t n_out = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_out, dest,
		[compare, a](const T data)
		{
			return (T) compare(a, data);
		});
	}),
	[compare, name](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// todo: consider correctness
		varptr<T> gradb = args[0].second;
		return conditional<T>(0, gradb, compare, name);
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

template <typename T>
varptr<T> conditional (const varptr<T> a, T b, std::function<bool(T,T)> compare, std::string name)
{
	if (nullptr == a.get()) return nullptr;
	if (constant<T>* aconst = dynamic_cast<constant<T>*>(a.get()))
	{
		std::vector<T> acvec = expose<T>(aconst);
		for (T& acv : acvec)
		{
			acv = (T) compare(acv, b);
		}
		return constant<T>::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "conditional_" << name << "_" << b;
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}

	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>(
	[compare, b](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		size_t n_out = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_out, dest,
		[compare, b](const T data)
		{
			return (T) compare(data, b);
		});
	}),
	[compare, name](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// todo: consider correctness
		varptr<T> grada = args[0].second;
		return conditional<T>(grada, 0, compare, name);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

template <typename T>
varptr<T> conditional (const varptr<T> a, const varptr<T> b, std::function<bool(T,T)> compare, std::string name)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	std::unordered_set<inode<T>*> audience;
	std::string opname = "conditional_" + name;
	if (a->find_audience(opname, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a && args[1] == b)
				return aud;
		}
	}
	constant<T>* aconst = dynamic_cast<constant<T>*>(a.get());
	constant<T>* bconst = dynamic_cast<constant<T>*>(b.get());
	if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(aconst);
		return conditional(outconst[0], b, compare, name);
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<T> outconst = expose<T>(bconst);
		return conditional(a, outconst[0], compare, name);
	}

	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	new transfer_func<T>(binary_elem_agg(
	AGGREGATE<T>([compare](T left, T right) -> T
	{
		return (T) compare(left, right);
	}))),
	[compare, name](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		assert(args.size() == 2);
		varptr<T> grada = args[0].second;
		varptr<T> gradb = args[1].second;
		// todo: consider correctness
		return conditional<T>(grada, gradb, compare, name);
	}, opname);
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
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	new transfer_func<T>(
	[a](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [a](const T data)
		{
			return a + data;
		});
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>(
	[b](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [b](const T data)
		{
			return data + b;
		});
	}),
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
	return add_helper<T>(a, b, "add", elementary_shaper,
	binary_elem_agg(
	AGGREGATE<T>([](const T left, const T right) -> T
	{
		return left + right;
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
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	new transfer_func<T>(
	[a](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [a](const T data)
		{
			return a - data;
		});
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>(
	[b](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [b](const T data)
		{
			return data - b;
		});
	}),
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
	return sub_helper<T>(a, b, "sub", elementary_shaper,
	binary_elem_agg(
	AGGREGATE<T>([](const T left, const T right) -> T
	{
		return left - right;
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
	elementary_shaper,
	new transfer_func<T>(
	[a](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [a](const T data)
		{
			return a * data;
		});
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a},
	elementary_shaper,
	new transfer_func<T>(
	[b](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [b](const T data)
		{
			return data * b;
		});
	}),
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
	return mul_helper<T>(a, b, "mul", elementary_shaper,
	binary_elem_agg(
	AGGREGATE<T>([](const T left, const T right) -> T
	{
		return left * right;
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
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	new transfer_func<T>(
	[a](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [a](const T data)
		{
			return a / data;
		});
	}),
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
	if (inode<T>* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr<T> out = immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	new transfer_func<T>(
	[b](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest, [b](const T data)
		{
			return data / b;
		});
	}),
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
	return div_helper<T>(a, b, "div", elementary_shaper,
	binary_elem_agg(
	AGGREGATE<T>([](const T left, const T right) -> T
	{
		return left / right;
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
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left + right;
	}), axis_a, false),
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
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left + right;
	}), axis_b, true),
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
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left - right;
	}), axis_a, false),
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
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left - right;
	}), axis_b, true),
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
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left * right;
	}), axis_a, false),
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
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left * right;
	}), axis_b, true),
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
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left / right;
	}), axis_a, false),
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
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE<T>([](T left, T right) -> T
	{
		return left / right;
	}), axis_b, true),
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
