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
	tensorshape trueshape;
	tensorshape lastshape;
	for (size_t i = 0, nshapes = shapes.size(); i < nshapes; ++i)
	{
		if (shapes[i].n_elems() == 1) continue;
		if (false == shapes[i].is_compatible_with(lastshape.shape_dimensions()))
		{
			std::stringstream ss;
			ss << "shape ";
			print_shape(shapes[i], ss);
			ss << " is incompatible with shape ";
			print_shape(lastshape, ss);
			throw std::runtime_error(ss.str());
		}
		lastshape = shapes[i];
		if (lastshape.is_grouped())
		{
			// there can only be 1 defined true shape
			assert(false == trueshape.is_part_defined());
			trueshape = lastshape;
		}
	}
	if (trueshape.is_fully_defined()) return trueshape.as_list();
	if (false == lastshape.is_part_defined()) return std::vector<size_t>{1};
	return lastshape;
}

template <typename T>
inline void elementary_check (const varptr<T>& a, const varptr<T>& b)
{
	if (a->good_status() && b->good_status())
		elementary_shaper({a->get_shape(), b->get_shape()});
}

template <typename T>
varptr<T> operator + (const varptr<T> a)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = +in[i];
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = -in[i];
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::sin(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::cos(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::tan(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = 1/std::sin(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = 1/std::cos(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::cos(in[i])/std::sin(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::exp(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::sqrt(in[i]);
		}
	},
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
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[scalar](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::pow(in[i], scalar);
		}
	},
	[scalar](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sqrt'(f(x)) = f'(x) * (scalar*f(x)^(scalar-1))
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return scalar * varptr<T>(grad) * pow(a, scalar-1);
	}, "sqrt");
}

template <typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max)
{
	if (nullptr == a) return nullptr;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[min, max](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			T v = in[i];
			if (min > v) v = min;
			else if (max < v) v = max;
			dest[i] = v;
		}
	},
	[min, max](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
		return varptr<T>(grad) * clip_val(a, min, max);
	}, "clip_val");
}

template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap)
{
	assert(cap > 0); // todo: maybe throw to indicate usage error
	if (nullptr == a) return nullptr;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[cap](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		const T* in = args.at(0);

		size_t ns = shape.n_elems();
		// calculate l2norm
		T l2norm = 0;
		for (size_t i = 0; i < ns; i++)
		{
			l2norm += in[i] * in[i];
		}
		l2norm = std::sqrt(l2norm);
		if (l2norm > cap)
		{
			// normalize
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = in[i] * cap / l2norm;
			}
			T newl2norm = 0;
			for (size_t i = 0; i < ns; i++)
			{
				newl2norm += dest[i] * dest[i];
			}
			newl2norm = std::sqrt(newl2norm);
		}
		else
		{
			std::memcpy(dest, in, sizeof(T) * ns);
		}
	},
	[cap](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		varptr<T> a = args.front();
		inode<T>* grad;
		a->get_leaf(grad, leaf);
	   	return varptr<T>(grad) * clip_norm(a, cap);
	}, "clip_norm");
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
	}
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a + bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = g'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return grad;
	}, "add");
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
	}
	if (b == (T)0) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] + b;
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return grad;
	}, "add");
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
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape,
		std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
	{
		size_t ns = shape.n_elems();
		tensorshape& ashape = inshapes[0];
		tensorshape& bshape = inshapes[1];
		const T* ad = args.at(0);
		const T* bd = args.at(1);

		if (ashape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[0] + bd[i];
			}
		}
		else if (bshape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] + bd[0];
			}
		}
		else if (ashape.is_grouped())
		{
			for (size_t i = 0, bn = bshape.n_elems(); i < bn; i++)
			{
				std::vector<size_t> aindices = ashape.memory_indices(i);
				T bval = bd[i] / aindices.size(); // spread b's value across all mapped a elements
				for (size_t aidx : aindices)
				{
					dest[aidx] = ad[aidx] + bval;
				}
			}
		}
		else if (bshape.is_grouped())
		{
			for (size_t i = 0, an = ashape.n_elems(); i < an; i++)
			{
				std::vector<size_t> bindices = bshape.memory_indices(i);
				T aval = ad[i] / bindices.size(); // spread a's value across all mapped b elements
				for (size_t bidx : bindices)
				{
					dest[bidx] = aval + bd[bidx];
				}
			}
		}
		else
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] + bd[i];
			}
		}
	},
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
	}
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a - bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -g'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return -varptr<T>(grad);
	}, "sub");
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
	}
	if (b == (T)0) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] - b;
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		inode<T>* grad;
		args.at(0)->get_leaf(grad, leaf);
		return varptr<T>(grad);
	}, "sub");
}

template <typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
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
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape,
		std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
	{
		size_t ns = shape.n_elems();
		tensorshape& ashape = inshapes[0];
		tensorshape& bshape = inshapes[1];
		const T* ad = args.at(0);
		const T* bd = args.at(1);

		if (ashape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[0] - bd[i];
			}
		}
		else if (bshape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] - bd[0];
			}
		}
		else if (ashape.is_grouped())
		{
			for (size_t i = 0, bn = bshape.n_elems(); i < bn; i++)
			{
				std::vector<size_t> aindices = ashape.memory_indices(i);
				T bval = bd[i] / aindices.size(); // spread b's value across all mapped a elements
				for (size_t aidx : aindices)
				{
					dest[aidx] = ad[aidx] - bval;
				}
			}
		}
		else if (bshape.is_grouped())
		{
			for (size_t i = 0, an = ashape.n_elems(); i < an; i++)
			{
				std::vector<size_t> bindices = bshape.memory_indices(i);
				T aval = ad[i] / bindices.size(); // spread a's value across all mapped b elements
				for (size_t bidx : bindices)
				{
					dest[bidx] = aval - bd[bidx];
				}
			}
		}
		else
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] - bd[i];
			}
		}
	},
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
	}
	if (0 == a)
	{
		return constant<T>::get(0);
	}
	if (1 == a) return b;
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a * bd[i];
		}
	},
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = c*g'(x)
		inode<T>* bg;
		args.at(0)->get_leaf(bg, leaf);
		return a * varptr<T>(bg);
	}, "mul");
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
	}
	if (0 == b) return constant<T>::get(0);
	if (1 == b) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] * b;
		}
	},
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = c*f'(x)
		inode<T>* ag;
		args.at(0)->get_leaf(ag, leaf);
		return b * varptr<T>(ag);
	}, "mul");
}

template <typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
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
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape,
	   std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
	{
		size_t ns = shape.n_elems();
		tensorshape& ashape = inshapes[0];
		tensorshape& bshape = inshapes[1];
		const T* ad = args.at(0);
		const T* bd = args.at(1);

		if (ashape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[0] * bd[i];
			}
		}
		else if (bshape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] * bd[0];
			}
		}
		else if (ashape.is_grouped())
		{
			for (size_t i = 0, bn = bshape.n_elems(); i < bn; i++)
			{
				std::vector<size_t> aindices = ashape.memory_indices(i);
				for (size_t aidx : aindices)
				{
					dest[aidx] = ad[aidx] * bd[i];
				}
			}
		}
		else if (bshape.is_grouped())
		{
			for (size_t i = 0, an = ashape.n_elems(); i < an; i++)
			{
				std::vector<size_t> bindices = bshape.memory_indices(i);
				for (size_t bidx : bindices)
				{
					dest[bidx] = ad[i] * bd[bidx];
				}
			}
		}
		else
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] * bd[i];
			}
		}
	},
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
	}
	if (a == (T)0)
	{
		return constant<T>::get(0);
	}
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a / bd[i];
		}
	},
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr<T> b = args.at(0);
		inode<T>* bg;
		b->get_leaf(bg, leaf);
		return -a * varptr<T>(bg) / (b * b);
	}, "div");
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
	}
	if (b == 0)
	{
		throw std::logic_error("divide by zero");
	}
	if (b == (T)1) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args.at(0);
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] / b;
		}
	},
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)/c
		inode<T>* ag;
		args.at(0)->get_leaf(ag, leaf);
		return varptr<T>(ag) / b;
	}, "div");
}

template <typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
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
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape,
		std::vector<const T*>& args, std::vector<tensorshape>& inshapes)
	{
		size_t ns = shape.n_elems();
		tensorshape& ashape = inshapes[0];
		tensorshape& bshape = inshapes[1];
		const T* ad = args.at(0);
		const T* bd = args.at(1);

		if (ashape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[0] / bd[i];
			}
		}
		else if (bshape.n_elems() == 1)
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] / bd[0];
			}
		}
		else if (ashape.is_grouped())
		{
			for (size_t i = 0, bn = bshape.n_elems(); i < bn; i++)
			{
				std::vector<size_t> aindices = ashape.memory_indices(i);
				for (size_t aidx : aindices)
				{
					dest[aidx] = ad[aidx] / bd[i];
				}
			}
		}
		else if (bshape.is_grouped())
		{
			for (size_t i = 0, an = ashape.n_elems(); i < an; i++)
			{
				std::vector<size_t> bindices = bshape.memory_indices(i);
				for (size_t bidx : bindices)
				{
					dest[bidx] = ad[i] / bd[bidx];
				}
			}
		}
		else
		{
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = ad[i] / bd[i];
			}
		}
	},
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
