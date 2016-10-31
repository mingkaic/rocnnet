//
//  elementary.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef elementary_hpp
#include <iostream>

namespace nnet {

// Elementary Operations

template <typename T>
void elementary<T>::make_gradient (VAR_PTR<T>& safety_ref) {
	this->set_gradient(der(args));
	safety_ref = this->grad;
}

template <typename T>
EVOKER_PTR<T> elementary<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new elementary(args, op, der, name));
}

template <typename T>
void elementary<T>::replace (ivariable<T>* food, VAR_PTR<T> newfood) {
	for (size_t i = 0; i < args.size(); i++) {
		if (args[i].get() == food) args[i] = newfood;
	}
}

template <typename T>
void elementary<T>::shape_eval (void) {
	auto it = args.begin();
	tensor_shape first = this->get_eval(*it).get_shape();
	if (first.is_fully_defined()) {
		for (it++; args.end() != it; it++) {
			tensor_shape ts = this->get_eval(*it).get_shape();
			assert(first.is_compatible_with(ts) ||
				   1 == ts.n_dims() || 1 == first.n_dims());
			if (ts.n_dims() > first.n_dims()) first = ts;
		}
		this->update(first);
	}
}

template <typename T>
elementary<T>::elementary (std::vector<VAR_PTR<T> > args,
std::function<void(T&, T)> op,
		ELEMENTARY_DERIV<T> der,
std::string name) : op(op), der(der), args(args) {
	this->name = name;
	for (VAR_PTR<T> a : args) {
		this->consume(*(a.get()));
	}
	if (session::pre_shape_eval()) {
		shape_eval();
	}
}

template <typename T>
const tensor<T>& elementary<T>::eval (void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	auto it = args.begin();
	if (1 == args.size()) {
		this-> out_ = tensor<T>(0);
	} else {
		this-> out_ = (*it)->eval();
		it++;
	}
	while (args.end() != it) {
		this->elem_op(this-> out_, (*it)->eval(), op);
		it++;
	}
	return this-> out_;
}

// nulls are treated as 0
template <typename T>
VAR_PTR<T> operator + (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = +other; },
	[](std::vector<VAR_PTR<T> > args) { return +args.front()->get_gradient(); },
	nnutils::formatter() << "abs(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> operator - (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = -other; },
	[](std::vector<VAR_PTR<T> > args) { return -args.front()->get_gradient(); },
	nnutils::formatter() << "neg(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> sin (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = std::sin(other); },
	[](std::vector<VAR_PTR<T> > args) {
		// sin'(f(x)) = f'(x)*cos(f(x))
		VAR_PTR<T> a = args.front();
		return a->get_gradient() * cos(a);
	}, nnutils::formatter() << "sin(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> cos (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = std::cos(other); },
	[](std::vector<VAR_PTR<T> > args) {
		// cos'(f(x)) = -f'(x)*sin(f(x))
		VAR_PTR<T> a = args.front();
		return -a->get_gradient() * sin(a);
	}, nnutils::formatter() << "cos(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> tan (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = std::tan(other); },
	[](std::vector<VAR_PTR<T> > args) {
		// sec'(f(x)) = f'(x)*sec^2(f(x))
		// better with = f'(x)/cos^2(f(x))
		VAR_PTR<T> a = args.front();
		VAR_PTR<T> denom = cos(a);
		return a->get_gradient() / (denom * denom);
 	}, nnutils::formatter() << "tan(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> csc (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = 1/std::sin(other); },
	[](std::vector<VAR_PTR<T> > args) {
		// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
		// better with -f'(x)/(sin(f(x)*tan(f(x))))
		VAR_PTR<T> a = args.front();
		return -a->get_gradient() / (sin(a) * tan(a));
	}, nnutils::formatter() << "csc(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> sec (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = 1/std::cos(other); },
	[](std::vector<VAR_PTR<T> > args) {
		// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
		// better with f'(x)*tan(f(x))/cos(f(x))
		VAR_PTR<T> a = args.front();
		return a->get_gradient() * tan(a) / cos(a);
	}, nnutils::formatter() << "sec(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> cot (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = 1/std::tan(other); },
	[](std::vector<VAR_PTR<T> > args) {
		// cot'(f(x)) = -f'(x)*csc^2(f(x))
		VAR_PTR<T> a = args.front();
		VAR_PTR<T> b = csc(a);
		return -a->get_gradient() * b * b;
	}, nnutils::formatter() << "cot(" << a->get_name() << ")");
	return op;
}

template <typename T>
VAR_PTR<T> exp (const VAR_PTR<T>& a) {
	if (nullptr == a) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a},
	[](T& collector, T other) { collector = std::exp(other); },
	[](std::vector<VAR_PTR<T> > args) {
		// exp'(f(x)) = f'(x)*exp(f(x))
		VAR_PTR<T> a = args.front();
		return a->get_gradient() * exp(a);
	}, nnutils::formatter() << "exp(" << a->get_name() << ")");
	return op;
}

template<typename T>
VAR_PTR<T> operator + (T a, const VAR_PTR<T>& b) {
	return constant<T>::make(a) + b;
}

template<typename T>
VAR_PTR<T> operator + (const VAR_PTR<T>& a, T b) {
	return a + constant<T>::make(b);
}

template <typename T>
VAR_PTR<T> operator + (const VAR_PTR<T>& a, const VAR_PTR<T>& b) {
	if (nullptr == a) return b;
	else if (nullptr == b) return a;

	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a, b},
	[](T& collector, T other) { collector += other; },
	[](std::vector<VAR_PTR<T> > args) {
		// h'(f(x), g(x)) = f'(x) + g'(x)
		auto it = args.begin();
		VAR_PTR<T> res = (*it)->get_gradient();
		for (it++; args.end() != it; it++) {
			res = res + (*it)->get_gradient();
		}
		return res;
	}, nnutils::formatter() << "(" << a->get_name() << "+" << b->get_name() << ")");
	return op;
}

template<typename T>
VAR_PTR<T> operator - (T a, const VAR_PTR<T>& b) {
	return constant<T>::make(a) - b;
}

template<typename T>
VAR_PTR<T> operator - (const VAR_PTR<T>& a, T b) {
	return a - constant<T>::make(b);
}

template <typename T>
VAR_PTR<T> operator - (const VAR_PTR<T>& a, const VAR_PTR<T>& b) {
	if (nullptr == a) return b;
	else if (nullptr == b) return a;

	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a, b},
	[](T& collector, T other) { collector -= other; },
	[](std::vector<VAR_PTR<T> > args) {
		// h'(f(x), g(x)) = f'(x) - g'(x)
		auto it = args.begin();
		VAR_PTR<T> res = (*it)->get_gradient();
		for (it++; args.end() != it; it++) {
			res = res - (*it)->get_gradient();
		}
		return res;
	}, nnutils::formatter() << "(" << a->get_name() << "-" << b->get_name() << ")");
	return op;
}

template<typename T>
VAR_PTR<T> operator * (T a, const VAR_PTR<T>& b) {
	return constant<T>::make(a) * b;
}

template<typename T>
VAR_PTR<T> operator * (const VAR_PTR<T>& a, T b) {
	return a * constant<T>::make(b);
}

template <typename T>
VAR_PTR<T> operator * (const VAR_PTR<T>& a, const VAR_PTR<T>& b) {
	if (nullptr == a || nullptr == b) return nullptr;
	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a, b},
	[](T& collector, T other) { collector *= other; },
	[](std::vector<VAR_PTR<T> > args) {
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		VAR_PTR<T> a = args.front();
		VAR_PTR<T> b = args.back();
		return a->get_gradient() * b + b->get_gradient() * a;
	}, nnutils::formatter() << a->get_name() << "*" << b->get_name());
	return op;
}

template<typename T>
VAR_PTR<T> operator / (T a, const VAR_PTR<T>& b) {
	return constant<T>::make(a) / b;
}

template<typename T>
VAR_PTR<T> operator / (const VAR_PTR<T>& a, T b) {
	return a / constant<T>::make(b);
}

template <typename T>
VAR_PTR<T> operator / (const VAR_PTR<T>& a, const VAR_PTR<T>& b) {
	if (nullptr == a) return nullptr;
	assert (nullptr != b); // don't allow infinity

	VAR_PTR<T> op = elementary<T>::make(std::vector<VAR_PTR<T> >{a, b},
	[](T& collector, T other) { collector /= other; },
	[](std::vector<VAR_PTR<T> > args) {
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		VAR_PTR<T> a = args.front();
		VAR_PTR<T> b = args.back();
		return (a->get_gradient() * b - b->get_gradient() * a) / (b * b);
	}, nnutils::formatter() << a->get_name() << "/" << b->get_name());
	return op;
}

}

#endif
