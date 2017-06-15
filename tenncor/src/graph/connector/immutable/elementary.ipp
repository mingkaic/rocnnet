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
		if (shapes[i].n_elems() == 1) continue;
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
inline void elementary_check (const varptr<T>& a, const varptr<T>& b)
{
	if (a->good_status() && b->good_status())
		elementary_shaper({a->get_shape(), b->get_shape()});
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

template <typename T>
inline transfer_func<T>* binary_axial_agg (ELEM_FUNC<T> aggregate, size_t axis)
{
	return new transfer_func<T>(
	[axis](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape lastshape = shapes[0];
		for (size_t i = 1, nshapes = shapes.size(); i < nshapes; ++i)
		{
			if (shapes[i].n_elems() == 1) continue;
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
	{
		[axis](size_t i, tensorshape& ashape, const tensorshape& outshape) -> std::vector<size_t>
		{
			if (1 == ashape.n_elems()) return {0};
			std::vector<size_t> alist = ashape.as_list();
			if (axis >= alist.size() || (alist[axis] == 1 && outshape.as_list()[axis] > 1))
			{
				std::vector<size_t> coord = outshape.coordinate_from_idx(i);
				coord[axis] = 0;
				i = ashape.sequential_idx(coord);
			}
			return {i};
		},
		[axis](size_t i, tensorshape& bshape, const tensorshape& outshape) -> std::vector<size_t>
		{
			if (1 == bshape.n_elems()) return {0};
			std::vector<size_t> blist = bshape.as_list();
			if (axis >= blist.size() || (blist[axis] == 1 && outshape.as_list()[axis] > 1))
			{
				std::vector<size_t> coord = outshape.coordinate_from_idx(i);
				coord[axis] = 0;
				i = bshape.sequential_idx(coord);
			}
			return {i};
		}
	}, aggregate);
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return +group[0];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* a1 = args.front();
		inode<T>* grad;
		a1->get_leaf(grad, leaf);
		return +varptr<T>(grad);
	}, "abs");
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
		std::vector<subject*> childargs = aconn->get_subjects();
		if (0 == a->get_label().compare("neg") && 1 == childargs.size())
		{
			// avoids double negatives by calling child directly
			return static_cast<inode<T>*>(childargs[0]);
		}
	}
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return -group[0];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* a1 = args.front();
		inode<T>* grad;
		a1->get_leaf(grad, leaf);
		return -varptr<T>(grad);
	}, "neg");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return std::sin(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sin'(f(x)) = f'(x)*cos(f(x))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return varptr<T>(grad) * cos(a);
	}, "sin");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return std::cos(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// cos'(f(x)) = -f'(x)*sin(f(x))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return -varptr<T>(grad) * sin(a);
	}, "cos");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return std::tan(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sec'(f(x)) = f'(x)*sec^2(f(x))
		// better with = f'(x)/cos^2(f(x))
		varptr<T> a = args.front();
		varptr<T> denom = cos(a);
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return varptr<T>(grad) / (denom * denom);
	}, "tan");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return 1/std::sin(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
		// better with -f'(x)/(sin(f(x)*tan(f(x))))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return -varptr<T>(grad) / (sin(a) * tan(a));
	}, "csc");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return 1/std::cos(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
		// better with f'(x)*tan(f(x))/cos(f(x))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return varptr<T>(grad) * tan(a) / cos(a);
	}, "sec");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return std::cos(group[0]) / std::sin(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// cot'(f(x)) = -f'(x)*csc^2(f(x))
		varptr<T> a = args.front();
		varptr<T> b = csc(a);
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return -varptr<T>(grad) * b * b;
	}, "cot");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return std::exp(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return varptr<T>(grad) * exp(a);
	}, "exp");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return std::sqrt(group[0]);
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sqrt'(f(x)) = f'(x)/(2*sqrt(f(x)))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return varptr<T>(grad) / ((T)2 * sqrt(a));
	}, "sqrt");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([scalar](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return std::pow(group[0], scalar);
	})),
	[scalar](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sqrt'(f(x)) = f'(x) * (scalar*f(x)^(scalar-1))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return scalar * varptr<T>(grad) * pow(a, scalar-1);
	}, nnutils::formatter() << "pow_" << scalar);
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([min, max](const T* group, size_t n) -> T
	{
		assert(n == 1);
		T v = group[0];
		// todo: we can make comparison slightly faster by xor min
		if (min > v) v = min;
		else if (max < v) v = max;
		return v;
	})),
	[min, max](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return varptr<T>(grad) * clip_val(a, min, max);
	}, nnutils::formatter() << "clip_val_" << min << "_" << max);
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	new transfer_func<T>(
	[](std::vector<tensorshape> inshapes) { return std::vector<size_t>{inshapes[0].rank()}; },
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
	[](const T* group, size_t n)
	{
		T l2norm = 0;
		for (size_t i = 0; i < n; i++)
		{
			l2norm += group[i] * group[i];
		}
		return std::sqrt(l2norm);
	}),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* grad;
		args[0]->get_leaf(grad, leaf);
		return l2norm(varptr<T>(grad));
	}, "l2norm");
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
	return immutable<T>::get(std::vector<inode<T>*>{l2norm(a), a},
	binary_elem_agg(ELEM_FUNC<T>([cap](const T* group, size_t n) -> T
	{
		assert(n == 2);
		T l2norm = group[0];
		T elem = group[1];
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
		inode<T>* grad;
		a->get_leaf(grad, leaf);
	   	return varptr<T>(grad) * clip_norm(a, cap);
	}, nnutils::formatter() << "clip_l2norm_" << cap);
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
	return immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T* group, size_t n) -> T
	 {
	 	assert(n == 1);
		return a + group[0];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = g'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return grad;
	}, nnutils::formatter() << a << "_add");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return group[0] + b;
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return grad;
	}, nnutils::formatter() << "add_" << b);
}

template <typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
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
	elementary_check(a, b);
	return immutable<T>::get(std::vector<inode<T>*>{a, b},
	binary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 2);
		return group[0] + group[1];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		inode<T>* ag;
		inode<T>* bg;
		args.at(0)->get_leaf(ag, leaf);
		args.at(1)->get_leaf(bg, leaf);
		return varptr<T>(ag) + varptr<T>(bg);
	}, "add");
}

template <typename T>
varptr<T> add (const varptr<T> a, const varptr<T> b, size_t axis)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	if (a.get() == b.get()) return (T)2 * a;
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
	return immutable<T>::get(std::vector<inode<T>*>{a, b},
	binary_axial_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 2);
		return group[0] + group[1];
	}), axis),
	[axis](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		inode<T>* ag;
		inode<T>* bg;
		args.at(0)->get_leaf(ag, leaf);
		args.at(1)->get_leaf(bg, leaf);
		return add(varptr<T>(ag), varptr<T>(bg), axis);
	}, nnutils::formatter() << "add_axis_" << axis);
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
	return immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return a - group[0];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -g'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return -varptr<T>(grad);
	}, nnutils::formatter() << a << "_sub");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return group[0] - b;
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return varptr<T>(grad);
	}, nnutils::formatter() << "sub_" << b);
}

template <typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
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
	elementary_check(a, b);
	return immutable<T>::get(std::vector<inode<T>*>{a, b},
	binary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 2);
		return group[0] - group[1];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		inode<T>* ag;
		inode<T>* bg;
		args.at(0)->get_leaf(ag, leaf);
		args.at(1)->get_leaf(bg, leaf);
		return varptr<T>(ag) - varptr<T>(bg);
	}, "sub");
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
	return immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return a * group[0];
	})),
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = c*g'(x)
		inode<T>* bg;
		args.at(0)->get_leaf(bg, leaf);
		return a * varptr<T>(bg);
	}, nnutils::formatter() << a << "_mul");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return group[0] * b;
	})),
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = c*f'(x)
		inode<T>* ag;
		args.at(0)->get_leaf(ag, leaf);
		return b * varptr<T>(ag);
	}, nnutils::formatter() << "mul_" << b);
}

template <typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
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
	elementary_check(a, b);
	return immutable<T>::get(std::vector<inode<T>*>{a, b},
	binary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 2);
		return group[0] * group[1];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		inode<T>* ag;
		inode<T>* bg;
		varptr<T> a = args.at(0);
		varptr<T> b = args.at(1);
		a->get_leaf(ag, leaf);
		b->get_leaf(bg, leaf);
		return varptr<T>(ag) * b + varptr<T>(bg) * a;
	}, "mul");
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
	return immutable<T>::get(std::vector<inode<T>*>{b},
	unary_elem_agg(ELEM_FUNC<T>([a](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return a / group[0];
	})),
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr<T> b = args.at(0);
		inode<T>* bg;
		b->get_leaf(bg, leaf);
		return -a * varptr<T>(bg) / (b * b);
	}, nnutils::formatter() << a << "_div");
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
	return immutable<T>::get(std::vector<inode<T>*>{a},
	unary_elem_agg(ELEM_FUNC<T>([b](const T* group, size_t n) -> T
	{
		assert(n == 1);
		return group[0] / b;
	})),
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)/c
		inode<T>* ag;
		args.at(0)->get_leaf(ag, leaf);
		return varptr<T>(ag) / b;
	}, nnutils::formatter() << "div_" << b);
}

template <typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
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
	elementary_check(a, b);
	return immutable<T>::get(std::vector<inode<T>*>{a, b},
	binary_elem_agg(ELEM_FUNC<T>([](const T* group, size_t n) -> T
	{
		assert(n == 2);
		return group[0] / group[1];
	})),
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		inode<T>* ag;
		inode<T>* bg;
		varptr<T> a = args.at(0);
		varptr<T> b = args.at(1);
		a->get_leaf(ag, leaf);
		b->get_leaf(bg, leaf);
		return (varptr<T>(ag) * b - varptr<T>(bg) * a) / (b * b);
	}, "div");
}

}

#endif
