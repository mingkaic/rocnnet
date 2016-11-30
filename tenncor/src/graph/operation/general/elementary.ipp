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
		if (ivariable<T>* arg = child->to_type<ivariable<T> >())
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
		if (ivariable<T>* v = sub->to_type<ivariable<T> >())
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
	der_(other.der_),
	tens_buffer_(other.tens_buffer_),
	ioperation<T>(other, name) {}

template <typename T>
ivariable<T>* elementary<T>::clone_impl (std::string name)
{
	return new elementary<T>(*this, name);
}

template <typename T>
elementary<T>::elementary (std::vector<ivariable<T>*> args,
	TEN_OP<T> op, BUILD_DERIVE<T> der,
	std::string name) :
	der_(der),
	ioperation<T>(args, name)
{
	this->out_ = std::make_unique<tensor_op<T> >(op,
	[](std::vector<tensorshape> shapes)
	{
		tensorshape first = std::vector<size_t>{1};
		for (tensorshape s : shapes)
		{
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
		return first;
	});
	// try to update
	update(ccoms::update_message(nullptr));
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
		der_ = other.der_;
		this->copy(other);
	}
	return *this;
}

template <typename T>
void elementary<T>::update (ccoms::update_message msg)
{
	static tensor<T> ones(1);

	// cast caller dependency as ivariable
	ivariable<T>* caller = nullptr;
	if (msg.caller_)
	{
		caller = msg.caller_->to_type<ivariable<T> >();
	}
	tensor<T>* storage = nullptr;
	this->valid_tensor_ = true;

	// if caller is null then update all tensors
	if (nullptr == caller)
	{
		tens_buffer_.clear();
		for (ccoms::subject* sub : this->dependencies_)
		{
			if (ivariable<T>* var = sub->to_type<ivariable<T> >())
			{
				storage = var->get_eval();
				tens_buffer_.push_back(storage);
				if (nullptr == storage)
				{
					this->valid_tensor_ = false;
				}
			}
		}
	}
	else
	{
		ivariable<T>* grad = nullptr;
		if (msg.grad_)
		{
			grad = msg.grad_->to_type<ivariable<T> >();
		}
		// determine caller index
		size_t idx = 0;
		for (ccoms::subject* sub : this->dependencies_)
		{
			if (sub->to_type<ivariable<T> >() == caller) break;
			idx++;
		}
		assert(idx < this->dependencies_.size()); // same as caller is in dependencies
		
		if (nullptr == grad) // don't care about grad, get best evaluation
		{
			storage = caller->get_eval();
			if (nullptr == storage)
			{
				this->valid_tensor_ = false;
			}
		}
		else if (ileaf<T>* leaf = dynamic_cast<ileaf<T>*>(grad)) // eval if caller is grad, null otherwise
		{
			storage = leaf == caller ? leaf->get_eval() : nullptr;
		}
		else
		{
			storage = grad == caller ? &ones : nullptr;
		}
		// update caller tensor only
		tens_buffer_[idx] = storage;
	}

	// tensor update when ready
	if (this->valid_tensor_)
	{
		// null is treated as erroneous zero
		(*this->out_)(tens_buffer_);
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = +in[i];
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = -in[i];
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::sin(in[i]);
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::cos(in[i]);
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::tan(in[i]);
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = 1/std::sin(in[i]);
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = 1/std::cos(in[i]);
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::cos(in[i])/std::sin(in[i]);
		}
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
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = std::exp(in[i]);
		}
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
	[min, max](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* in = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			T v = in[i];
			if (min > v) v = min;
			else if (max < v) v = max;
			dest[i] = v;
		}
	},
	[min, max](std::vector<ivariable<T>*> args)
	{
		varptr<T> a = args.front();
		return clip_val(varptr<T>(a->get_gradient()), min, max);
	},
	"clip_val(" + a->get_name() + ")");
	return op;
}

template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[cap](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		const T* in = args[0];

		size_t ns = info.res_shape_.n_elems();
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
	[cap](std::vector<ivariable<T>*> args)
	{
	   ivariable<T>* a = args.front();
	   return clip_norm(varptr<T>(a->get_gradient()), cap);
	},
	"clip_norm(" + a->get_name() + ")");
	return op;
}

template<typename T>
varptr<T> operator + (T a, const varptr<T> b)
{
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (nullptr == (ivariable<T>*)b) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{b},
	[a](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a + bd[i];
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(c, g(x)) = g'(x)
		ivariable<T>* b = args[0];
		return b->get_gradient();
	},
	nnutils::formatter() << "(" << a << "+" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator + (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] + b;
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = f'(x)
		ivariable<T>* a = args[0];
		return a->get_gradient();
	},
	nnutils::formatter() << "(" << a->get_name() << "+" << b << ")");
	return op;
}

// TRUE BINARY OPERATION
template <typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a) return b;
	else if (nullptr == (ivariable<T>*)b) return a;

	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		size_t as = info.arg_shape_[0].n_elems();
		size_t bs = info.arg_shape_[1].n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			T scalar_a = 1 == as ? ad[0] : ad[i];
			T scalar_b = 1 == bs ? bd[0] : bd[i];
			dest[i] = scalar_a + scalar_b;
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		ivariable<T>* a = args[0];
		ivariable<T>* b = args[1];
		varptr<T> ag = a->get_gradient();
		varptr<T> bg = b->get_gradient();
		return ag + bg;
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
	[a](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a - bd[i];
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(c, g(x)) = -g'(x)
		ivariable<T>* b = args[0];
		return -varptr<T>(b->get_gradient());
	},
	nnutils::formatter() << "(" << a << "-" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator - (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] - b;
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = f'(x)
		ivariable<T>* a = args[0];
		return a->get_gradient();
	},
	nnutils::formatter() << "(" << a->get_name() << "-" << b << ")");
	return op;
}

// TRUE BINARY OPERATION
template <typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a) return b;
	else if (nullptr == (ivariable<T>*)b) return a;

	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		size_t as = info.arg_shape_[0].n_elems();
		size_t bs = info.arg_shape_[1].n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			T scalar_a = 1 == as ? ad[0] : ad[i];
			T scalar_b = 1 == bs ? bd[0] : bd[i];
			dest[i] = scalar_a - scalar_b;
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		ivariable<T>* a = args[0];
		ivariable<T>* b = args[1];
		varptr<T> ag = a->get_gradient();
		varptr<T> bg = b->get_gradient();
		return ag - bg;
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
	[a](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a * bd[i];
		}
	},
	[a](std::vector<ivariable<T>*> args)
	{
		// h'(c, g(x)) = c*g'(x)
		varptr<T> bg = args[0]->get_gradient();
		return a * bg;
	},
	nnutils::formatter() << "(" << a << "*" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator * (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] * b;
		}
	},
	[b](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = c*f'(x)
		varptr<T> ag = args[0]->get_gradient();
		return b * ag;
	},
	nnutils::formatter() << "(" << a->get_name() << "*" << b << ")");
	return op;
}

// TRUE BINARY OPERATION
template <typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a || nullptr == (ivariable<T>*)b) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		size_t as = info.arg_shape_[0].n_elems();
		size_t bs = info.arg_shape_[1].n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			T scalar_a = 1 == as ? ad[0] : ad[i];
			T scalar_b = 1 == bs ? bd[0] : bd[i];
			dest[i] = scalar_a * scalar_b;
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr<T> a = args[0];
		varptr<T> b = args[1];
		varptr<T> ag = a->get_gradient();
		varptr<T> bg = b->get_gradient();
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
	[a](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* bd = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = a / bd[i];
		}
	},
	[a](std::vector<ivariable<T>*> args)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr<T> b = args[0];
		varptr<T> bg = b->get_gradient();
		return -a * bg / (b * b);
	},
	nnutils::formatter() << "(" << a << "/" << b->get_name() << ")");
	return op;
}

template<typename T>
varptr<T> operator / (const varptr<T> a, T b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a},
	[b](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		const T* ad = args[0];
		for (size_t i = 0; i < ns; i++)
		{
			dest[i] = ad[i] / b;
		}
	},
	[b](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), c) = f'(x)/c
		varptr<T> ag = args[0]->get_gradient();
		return ag / b;
	},
	nnutils::formatter() << "(" << a->get_name() << "/" << b << ")");
	return op;
}

// TRUE BINARY OPERATION
template <typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == (ivariable<T>*)a) return nullptr;
	assert (nullptr != (ivariable<T>*)b); // don't allow infinity

	ivariable<T>* op = elementary<T>::build(std::vector<ivariable<T>*>{a, b},
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		size_t ns = info.res_shape_.n_elems();
		size_t as = info.arg_shape_[0].n_elems();
		size_t bs = info.arg_shape_[1].n_elems();
		const T* ad = args[0];
		const T* bd = args[1];
		for (size_t i = 0; i < ns; i++)
		{
			T scalar_a = 1 == as ? ad[0] : ad[i];
			T scalar_b = 1 == bs ? bd[0] : bd[i];
			dest[i] = scalar_a / scalar_b;
		}
	},
	[](std::vector<ivariable<T>*> args)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr<T> a = args[0];
		varptr<T> b = args[1];
		varptr<T> ag = a->get_gradient();
		varptr<T> bg = b->get_gradient();
		return (ag * b - bg * a) / (b * b);
	},
	"(" + a->get_name() + "/" + b->get_name() + ")");
	return op;
}

}

#endif
