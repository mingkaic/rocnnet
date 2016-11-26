//
//  elementary.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef elementary_hpp

namespace nnet
{

// Elementary Operations

template <typename T>
void elementary<T>::setup_gradient (void)
{
	std::vector<ivariable<T>*> args;
	for (ccoms::subject* child : this->dependencies_)
	{
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child))
		{
			args.push_back(arg);
		}
	}
	this->grad_ = der_(args);
}

template <typename T>
tensorshape elementary<T>::shape_eval (void)
{
	tensorshape first = std::vector<size_t>{1};
	for (ccoms::subject* sub : this->dependencies_)
	{
		if (ivariable<T>* v = dynamic_cast<ivariable<T>*>(sub))
		{
			tensorshape s = v->get_shape();
			if (1 != first.n_dims() && 1 != s.n_dims() &&
				false == first.is_compatible_with(s))
			{
				throw std::invalid_argument(
					"cannot element-wise operate on tensors of vastly different shapes");
			}
			if (s.n_dims() >= first.n_dims())
			{
				first = s;
			}
		}
	}
	return first;
}

template <typename T>
elementary<T>::elementary (const elementary<T>& other, std::string name) :
	for_each_(other.for_each_),
	der_(other.der_),
	ioperation<T>(other, name),
	ivariable<T>(other, name),
	ccoms::iobserver(other) {}

template <typename T>
ivariable<T>* elementary<T>::clone_impl (std::string name)
{
	return new elementary<T>(*this, name);
}

template <typename T>
elementary<T>::elementary (std::vector<ivariable<T>*> args, 
	std::function<void(T&, T)> op, BUILD_DERIVE<T> der,
	std::string name) :
	for_each_(op),
	der_(der),
	ioperation<T>(args, name),
	ivariable<T>(std::vector<size_t>{}, name),
	ccoms::iobserver(std::vector<ccoms::subject*>(args.begin(), args.end()))
{
	// TODO: simplify operation arguments
	// TODO: no need to call shape_eval twice
	this->out_ = std::make_unique<tensor_op<T> >(
	[this](T* dest, std::vector<const T*> srcs)
	{
		tensorshape ts = shape_eval(); // call 1

		for (size_t i = 0; i < ts.n_elems(); i++)
		{
			auto it = srcs.begin();
			if (1 == srcs.size())
			{
				dest[i] = 0;
			}
			else
			{
				// n-nary operator. init with first object
				dest[i] = (*it)[i];
				it++;
			}
			while (srcs.end() != it)
			{
				for_each_(dest[i], (*it)[i]);
				it++;
			}
		}
	});
	// try to update
	update(nullptr);
	if (session::pre_shape_eval())
	{
		shape_eval();
	}
}

template <typename T>
elementary<T>* elementary<T>::clone (std::string name)
{
	return static_cast<elementary<T>*>(clone_impl(name));
}

template <typename T>
elementary<T>& elementary<T>::operator = (const elementary<T>& other)
{
	if (this != &other)
	{
		for_each_ = other.for_each_;
		der_ = other.der_;
		this->copy(other);
	}
	return *this;
}

template <typename T>
void elementary<T>::update (ccoms::subject* caller)
{
	tensor<T> one(1);
	std::vector<tensor<T>*> tens;
	this->valid_tensor_ = true;
	for (ccoms::subject* sub : this->dependencies_)
	{
		if (ivariable<T>* var = dynamic_cast<ivariable<T>*>(sub))
		{
			tensor<T>* a;
			if (nullptr == caller) 
			{
				a = var->get_eval();
				if (nullptr == a)
				{
					this->valid_tensor_ = false;
					break;
				}
			}
			else
			{
				a = var == caller ? &one : nullptr;
			}
			// a is var's eval if caller is nullptr, otherwise
			// a is one if var is the caller, nullptr otherwise
			tens.push_back(a);
		}
	}

	if (!this->out_->is_alloc())
	{
		this->out_->set_shape(shape_eval()); // call 2
	}
	// tensor update
	if (this->valid_tensor_)
	{
		(*this->out_)(tens);
	}

	this->notify();
}

// ELEMENTARY OPERATIONS

// nulls are treated as 0
template <typename T>
varptr<T> operator + (const varptr<T> a)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	return elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) 
		{
		    collector = +other;
	    },
		[](std::vector<ivariable<T>*> args) {
			varptr<T> grad = args.front()->get_gradient(); // wrap
			return +grad;
		},
	"abs(" + a->get_name() + ")");
}

template <typename T>
varptr<T> operator - (const varptr<T> a)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	return elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) 
		{
			collector = -other;
		},
		[](std::vector<ivariable<T>*> args)
		{
			varptr<T> grad = args.front()->get_gradient(); // wrap
			return -grad;
		},
	"neg(" + a->get_name() + ")");
}

template <typename T>
varptr<T> sin (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) {
			collector = std::sin(other);
		},
		[](std::vector<ivariable<T>*> args)
		{
			// sin'(f(x)) = f'(x)*cos(f(x))
			varptr<T> a = args.front();
			varptr<T> grad = a->get_gradient(); // wrap
			return grad * cos(a);
		}, 
	"sin(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> cos (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other)
		{
			collector = std::cos(other);
		},
		[](std::vector<ivariable<T>*> args)
		{
			// cos'(f(x)) = -f'(x)*sin(f(x))
			varptr<T> a = args.front();
			varptr<T> grad = a->get_gradient(); // wrap
			return -grad * sin(a);
		}, 
	"cos(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> tan (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other)
		{
			collector = std::tan(other);
		},
		[](std::vector<ivariable<T>*> args)
		{
			// sec'(f(x)) = f'(x)*sec^2(f(x))
			// better with = f'(x)/cos^2(f(x))
			varptr<T> a = args.front();
			varptr<T> denom = cos(a);
			varptr<T> grad = a->get_gradient(); // wrap
			return grad / (denom * denom);
	 	},
 	"tan(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> csc (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) 
		{
			collector = 1/std::sin(other);
		},
		[](std::vector<ivariable<T>*> args)
		{
			// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
			// better with -f'(x)/(sin(f(x)*tan(f(x))))
			varptr<T> a = args.front();
			varptr<T> grad = a->get_gradient(); // wrap
			return -grad / (sin(a) * tan(a));
		}, 
	"csc(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> sec (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other)
		{
			collector = 1/std::cos(other);
		},
		[](std::vector<ivariable<T>*> args)
		{
			// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
			// better with f'(x)*tan(f(x))/cos(f(x))
			varptr<T> a = args.front();
			varptr<T> grad = a->get_gradient(); // wrap
			return grad * tan(a) / cos(a);
		}, 
	"sec(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> cot (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other)
		{
			collector = std::cos(other)/std::sin(other);
		},
		[](std::vector<ivariable<T>*> args)
		{
			// cot'(f(x)) = -f'(x)*csc^2(f(x))
			varptr<T> a = args.front();
			varptr<T> b = csc(a);
			varptr<T> grad = a->get_gradient(); // wrap
			return -grad * b * b;
		}, 
	"cot(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> exp (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[](T& collector, T other)
		{
			collector = std::exp(other);
		},
		[](std::vector<ivariable<T>*> args)
		{
			// exp'(f(x)) = f'(x)*exp(f(x))
			varptr<T> a = args.front();
			varptr<T> grad = a->get_gradient(); // wrap
			return grad * exp(a);
		}, 
	"exp(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
		[min, max](T& collector, T other)
		{
			if (min > other) other = min;
			else if (max < other) other = max;
			collector = other;
		},
		[min, max](std::vector<ivariable<T>*> args)
		{
			varptr<T> a = args.front();
			return clip_val(varptr<T>(a->get_gradient()), min, max);
		}, 
	"clip_val(" + a->get_name() + ")");
	return op;
}

template<typename T>
varptr<T> operator + (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (nullptr == (ivariable<T>*)b) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{b},
	[a](T& collector, T other)
	{
	  collector = other + a;
	},
	[](std::vector<ivariable<T>*> args)
	{
	  // h'(c, g(x)) = g'(x)
		varptr<T> bx = args.back();
	  return bx->get_gradient();
	},
	nnutils::formatter() << "(" << a << "+" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator + (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](T& collector, T other)
	{
		collector = other + b;
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = f'(x)
		varptr<T> ax = args.front();
		return ax->get_gradient();
	},
	nnutils::formatter() << "(" << a->get_name() << "+" << b << ")");
	return op;
}

template <typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a) return b;
	else if (nullptr == (ivariable<T>*)b) return a;

	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other)
		{
			collector += other;
		},
		[](std::vector<ivariable<T>*> args)
		{
			// h'(f(x), g(x)) = f'(x) + g'(x)
			auto it = args.begin();
			varptr<T> res = (*it)->get_gradient();
			for (it++; args.end() != it; it++) {
				varptr<T> grad = (*it)->get_gradient();
				res = res + grad;
			}
			return res;
		}, 
	"(" + a->get_name() + "+" + b->get_name() + ")");
	return op;
}

template<typename T>
varptr<T> operator - (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (nullptr == (ivariable<T>*)b) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{b},
	[a](T& collector, T other)
	{
		collector = a - other;
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(c, g(x)) = -g'(x)
		varptr<T> bx = args.back();
		return -varptr<T>(bx->get_gradient());
	},
	nnutils::formatter() << "(" << a << "-" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator - (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](T& collector, T other)
	{
		collector = other - b;
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = f'(x)
		varptr<T> ax = args.front();
		return ax->get_gradient();
	},
	nnutils::formatter() << "(" << a->get_name() << "-" << b << ")");
	return op;
}

template <typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a) return b;
	else if (nullptr == (ivariable<T>*)b) return a;

	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other)
		{
			collector -= other;
		},
		[](std::vector<ivariable<T>*> args)
		{
			// h'(f(x), g(x)) = f'(x) - g'(x)
			auto it = args.begin();
			varptr<T> res = (*it)->get_gradient();
			for (it++; args.end() != it; it++) {
				varptr<T> grad = (*it)->get_gradient();
				res = res - grad;
			}
			return res;
		}, 
	"(" + a->get_name() + "-" + b->get_name() + ")");
	return op;
}

template<typename T>
varptr<T> operator * (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (nullptr == (ivariable<T>*)b) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{b},
	[a](T& collector, T other)
	{
		collector = other * a;
	},
	[a](std::vector<ivariable<T>*> args)
	{
		// h'(c, g(x)) = c*g'(x)
		varptr<T> bx = args.back();
		return a * varptr<T>(bx->get_gradient());
	},
	nnutils::formatter() << "(" << a << "*" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator * (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](T& collector, T other)
	{
		collector = other * b;
	},
	[b](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = c*f'(x)
		varptr<T> ax = args.front();
		return b * varptr<T>(ax->get_gradient());
	},
	nnutils::formatter() << "(" << a->get_name() << "*" << b << ")");
	return op;
}

template <typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a || nullptr == (ivariable<T>*)b) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other)
		{
			collector *= other;
		},
		[](std::vector<ivariable<T>*> args)
		{
			// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
			varptr<T> a = args.front();
			varptr<T> b = args.back();
			varptr<T> ag = a->get_gradient();
			varptr<T> bg = a->get_gradient();
			return ag * b + bg * a;
		},
	"(" + a->get_name() + "*" + b->get_name() + ")");
	return op;
}

template<typename T>
varptr<T> operator / (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (nullptr == (ivariable<T>*)b) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{b},
	[a](T& collector, T other)
	{
		collector = a / other;
	},
	[a](std::vector<ivariable<T>*> args)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr<T> bx = args.back();
		return -a * varptr<T>(bx->get_gradient()) / (bx * bx);
	},
	nnutils::formatter() << "(" << a << "/" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator / (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](T& collector, T other)
	{
		collector = other / b;
	},
	[b](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = f'(x)/c
		varptr<T> ax = args.front();
		return varptr<T>(ax->get_gradient()) / b;
	},
	nnutils::formatter() << "(" << a->get_name() << "/" << b << ")");
	return op;
}

template <typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	assert (nullptr != (ivariable<T>*)b); // don't allow infinity

	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other)
		{
			collector /= other;
		},
		[](std::vector<ivariable<T>*> args)
		{
			// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
			varptr<T> a = args.front();
			varptr<T> b = args.back();
			varptr<T> ag = a->get_gradient();
			varptr<T> bg = a->get_gradient();
			return (ag * b - bg * a) / (b * b);
		},
	"(" + a->get_name() + "/" + b->get_name() + ")");
	return op;
}

}

#endif
