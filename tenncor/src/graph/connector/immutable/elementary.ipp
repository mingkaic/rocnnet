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

tensorshape elementary_shaper (std::vector<tensorshape> shapes)
{
	tensorshape firstshape = shapes.front();
	size_t nshapes = shapes.size();
	for (size_t i = 1; i < nshapes; ++i)
	{
		if (false == shapes[i].is_compatible_with(firstshape))
		{
			throw std::exception(); // TODO: make better exception
		}
	}
	return firstshape;
}

// nulls are treated as 0
template <typename T>
varptr<T> operator + (const varptr<T> a)
{
	if (nullptr == (inode<T>*)a) return nullptr;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = +in[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* a1 = args.front();
		varptr<T> grad = a1->get_leaf(leaf); // wrap
		return +grad;
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
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = -in[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		inode<T>* a1 = args.front();
		varptr<T> grad = a1->get_leaf(leaf); // wrap
		return -grad;
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
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::sin(in[i]);
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// sin'(f(x)) = f'(x)*cos(f(x))
		varptr<T> a = args.front();
		varptr<T> grad = a->get_leaf(leaf); // wrap
		return grad * cos(a);
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
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::cos(in[i]);
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// cos'(f(x)) = -f'(x)*sin(f(x))
		varptr<T> a = args.front();
		varptr<T> grad = a->get_leaf(leaf); // wrap
		return -grad * sin(a);
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
		const T* in = args[0];
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
		varptr<T> grad = a->get_leaf(leaf); // wrap
		return grad / (denom * denom);
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
		const T* in = args[0];
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
		varptr<T> grad = a->get_leaf(leaf); // wrap
		return -grad / (sin(a) * tan(a));
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
		const T* in = args[0];
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
		varptr<T> grad = a->get_leaf(leaf); // wrap
		return grad * tan(a) / cos(a);
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
		const T* in = args[0];
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
		varptr<T> grad = a->get_leaf(leaf); // wrap
		return -grad * b * b;
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
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::exp(in[i]);
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr<T> a = args.front();
		varptr<T> grad = a->get_leaf(leaf); // wrap
		return grad * exp(a);
	}, "exp");
}

template <typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max)
{
	if (nullptr == a) return nullptr;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[min, max](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* in = args[0];
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
		varptr<T> grad = a->get_leaf(leaf);
		return clip_val(grad, min, max);
	}, "clip_val");
}

template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap)
{
	if (nullptr == a) return nullptr;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[cap](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		const T* in = args[0];

		size_t ns = shape.n_elems();
		// calculate l2norm
		T l2norm = 0;
		for (size_t i = 0; i < ns; i++)
		{
			l2norm += in[i] * in[i];
		}
		l2norm = std::sqrt(l2norm);
		// clip
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = in[i] * cap / l2norm;
		}
	},
	[cap](std::vector<inode<T>*> args, variable<T>* leaf)
	{
	   	inode<T>* a = args.front();
		varptr<T> grad = a->get_leaf(leaf);
	   	return clip_norm(varptr<T>(grad), cap);
	}, "clip_norm");
}

template<typename T>
varptr<T> operator + (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (*b == (T) 0 || nullptr == (inode<T>*)b) return constant<T>::get(a);
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a + bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = g'(x)
		return args[0]->get_leaf(leaf);
	}, "add");
}

template<typename T>
varptr<T> operator + (const varptr<T> a, T b)
{
	if (*a == (T) 0 || nullptr == (inode<T>*)a) return constant<T>::get(b);
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] + b;
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		return args[0]->get_leaf(leaf);
	}, "add");
}

template <typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b)
{
	if (*a == (T) 0 || nullptr == (inode<T>*)a) return b;
	else if (*b == (T) 0 || nullptr == (inode<T>*)b) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] + bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		varptr<T> ag = args[0]->get_leaf(leaf);
		varptr<T> bg = args[1]->get_leaf(leaf);
		return ag + bg;
	}, "add");
}

template<typename T>
varptr<T> operator - (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (*b == (T) 0 || nullptr == (inode<T>*)b) return constant<T>::get(a);
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a - bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -g'(x)
		return -varptr<T>(args[0]->get_leaf(leaf));
	}, "sub");
}

template<typename T>
varptr<T> operator - (const varptr<T> a, T b)
{
	if (*a == (T) 0 || nullptr == (inode<T>*)a) return constant<T>::get(-b);
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] - b;
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)
		return args[0]->get_leaf(leaf);
	}, "sub");
}

template <typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b)
{
	if (*a == (T) 0 || nullptr == (inode<T>*)a) return b;
	else if (*b == (T) 0 || nullptr == (inode<T>*)b) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] - bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		varptr<T> ag = args[0]->get_leaf(leaf);
		varptr<T> bg = args[1]->get_leaf(leaf);
		return ag - bg;
	}, "sub");
}

template<typename T>
varptr<T> operator * (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (*b == (T) 0 || nullptr == (inode<T>*)b || 0 == a) return constant<T>::get(0);
	if (*b == (T) 1) return constant<T>::get(a);
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a * bd[i];
		}
	},
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = c*g'(x)
		varptr<T> bg = args[0]->get_leaf(leaf);
		return a * bg;
	}, "mul");
}

template<typename T>
varptr<T> operator * (const varptr<T> a, T b)
{
	if (*a == (T) 0 || nullptr == (inode<T>*)a || 0 == b) return constant<T>::get(0);
	if (*a == (T) 1) return constant<T>::get(b);
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] * b;
		}
	},
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = c*f'(x)
		varptr<T> ag = args[0]->get_leaf(leaf);
		return b * ag;
	}, "mul");
}

template <typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b)
{
	if (*a == (T) 0 || *b == (T) 0 || nullptr == (inode<T>*)a || nullptr == (inode<T>*)b) return nullptr;
	if (*a == (T) 1) return b;
	if (*b == (T) 1) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] * bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr<T> a = args[0];
		varptr<T> b = args[1];
		varptr<T> ag = a->get_leaf(leaf);
		varptr<T> bg = b->get_leaf(leaf);
		return ag * b + bg * a;
	}, "mul");
}

template<typename T>
varptr<T> operator / (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	assert(*b != (T) 0 && nullptr != (inode<T>*) b);
	if (a == (T) 0) return constant<T>::get(0);
	if (*b == (T) 1) return constant<T>::get(a);
	return immutable<T>::get(std::vector<inode<T>*>{b}, elementary_shaper,
	[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a / bd[i];
		}
	},
	[a](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr<T> b = args[1];
		varptr<T> bg = b->get_leaf(leaf);
		return -a * bg / (b * b);
	}, "div");
}

template<typename T>
varptr<T> operator / (const varptr<T> a, T b)
{
	assert(b != 0);
	if (*a == (T) 0 || nullptr == (inode<T>*)a) return constant<T>::get(0);
	if (b == (T) 1) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a}, elementary_shaper,
	[b](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] / b;
		}
	},
	[b](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), c) = f'(x)/c
		varptr<T> ag = args[0]->get_leaf(leaf);
		return ag / b;
	}, "div");
}

template <typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b)
{
	assert (*b != (T) 0 && nullptr != (inode<T>*)b); // don't allow infinity
	if (*a == (T) 0 || nullptr == (inode<T>*)a) return constant<T>::get(0);
	if (*b == (T) 1) return a;
	return immutable<T>::get(std::vector<inode<T>*>{a, b}, elementary_shaper,
	[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] / bd[i];
		}
	},
	[](std::vector<inode<T>*> args, variable<T>* leaf)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr<T> a = args[0];
		varptr<T> b = args[1];
		varptr<T> ag = a->get_leaf(leaf);
		varptr<T> bg = b->get_leaf(leaf);
		return (ag * b - bg * a) / (b * b);
	}, "div");
}

}

#endif
