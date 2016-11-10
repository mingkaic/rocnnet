//
//  elementary.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef elementary_hpp

namespace nnet {

// Elementary Operations

template <typename T>
void elementary<T>::setup_gradient (void) {
	std::vector<ivariable<T>*> args;
	for (ccoms::subject* child : this->dependencies_) {
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
			args.push_back(arg);
		}
	}
	this->grad = der_(args);
}

template <typename T>
ievoker<T>* elementary<T>::clone_impl (std::string name) {
	return new elementary(std::vector<ivariable<T>*>(
        this->dependencies_.begin(), this->dependencies_.end()),
        for_each_, der_, name);
}

template <typename T>
void elementary<T>::shape_eval (void) {
	auto it = this->dependencies_.begin();
    ivariable<T>* arg = dynamic_cast<ivariable<T>*>(*it);
    assert(arg);
	tensor_shape first = this->get_tensor_from(arg).get_shape();
	if (first.is_fully_defined()) {
		for (it++; this->dependencies_.end() != it; it++) {
		    if ((arg = dynamic_cast<ivariable<T>*>(*it))) {
                tensor_shape ts = this->get_tensor_from(*it).get_shape();
                assert(first.is_compatible_with(ts) ||
                       1 == ts.n_dims() || 1 == first.n_dims());
                if (ts.n_dims() > first.n_dims()) first = ts;
			}
		}
		this->update(first);
	}
}

template <typename T>
elementary<T>::elementary (std::vector<ivariable<T>*> args,
		std::function<void(T&, T)> op,
		BUILD_DERIVE<T> der,
		std::string name) : 
			ioperation<T>(std::vector<ccoms::subject*>(args.begin(), args.end()), name),
			for_each_(op), 
			der_(der) {
	if (session::pre_shape_eval()) {
		shape_eval();
	}
}

template <typename T>
void elementary<T>::update (void) {
	auto it = this->dependencies_.begin();
	if (1 == this->dependencies_.size()) {
		// unary operator. init at zero
		this->out_ = tensor<T>(0);
	} else {
		// n-nary operator. init with first object
        ivariable<T>* arg = dynamic_cast<ivariable<T>*>(*it);
		this->out_ = arg->get_eval();
		it++;
	}

	while (this->dependencies_.end() != it) {
	    if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(*it)) {
		    this->elem_op(this->out_, arg->get_eval(), for_each_);
		}
		it++;
	}
	this->notify();
}

// ELEMENTARY OPERATIONS

// nulls are treated as 0
template <typename T>
ivariable<T>* operator + (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T> op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = +other; },
		[](std::vector<ivariable<T>*> args) {
			return +(args.front()->get_gradient()); 
		},
	nnutils::formatter() << "abs(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* operator - (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = -other; },
		[](std::vector<ivariable<T>*> args) {
			return -(args.front()->get_gradient());
		},
	nnutils::formatter() << "neg(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* sin (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = std::sin(other); },
		[](std::vector<ivariable<T>*> args) {
			// sin'(f(x)) = f'(x)*cos(f(x))
			ivariable<T>* a = args.front();
			return a->get_gradient() * cos(a);
		}, 
	nnutils::formatter() << "sin(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* cos (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = std::cos(other); },
		[](std::vector<ivariable<T>*> args) {
			// cos'(f(x)) = -f'(x)*sin(f(x))
			ivariable<T>* a = args.front();
			return -(a->get_gradient()) * sin(a);
		}, 
	nnutils::formatter() << "cos(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* tan (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = std::tan(other); },
		[](std::vector<ivariable<T>*> args) {
			// sec'(f(x)) = f'(x)*sec^2(f(x))
			// better with = f'(x)/cos^2(f(x))
			ivariable<T>* a = args.front();
			ivariable<T>* denom = cos(a);
			return a->get_gradient() / (denom * denom);
	 	},
 	nnutils::formatter() << "tan(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* csc (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = 1/std::sin(other); },
		[](std::vector<ivariable<T>*> args) {
			// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
			// better with -f'(x)/(sin(f(x)*tan(f(x))))
			ivariable<T>* a = args.front();
			return -a->get_gradient() / (sin(a) * tan(a));
		}, 
	nnutils::formatter() << "csc(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* sec (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = 1/std::cos(other); },
		[](std::vector<ivariable<T>*> args) {
			// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
			// better with f'(x)*tan(f(x))/cos(f(x))
			ivariable<T>* a = args.front();
			return a->get_gradient() * tan(a) / cos(a);
		}, 
	nnutils::formatter() << "sec(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* cot (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = 1/std::tan(other); },
		[](std::vector<ivariable<T>*> args) {
			// cot'(f(x)) = -f'(x)*csc^2(f(x))
			ivariable<T>* a = args.front();
			ivariable<T>* b = csc(a);
			return -a->get_gradient() * b * b;
		}, 
	nnutils::formatter() << "cot(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* exp (const ivariable<T>* a) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[](T& collector, T other) { collector = std::exp(other); },
		[](std::vector<ivariable<T>*> args) {
			// exp'(f(x)) = f'(x)*exp(f(x))
			ivariable<T>* a = args.front();
			return a->get_gradient() * exp(a);
		}, 
	nnutils::formatter() << "exp(" << a->get_name() << ")");
	return op;
}

template <typename T>
ivariable<T>* clip_by_value (const ivariable<T>* a, T min, T max) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[min, max](T& collector, T other) {
			if (min > other) other = min;
			else if (max < other) other = max;
			collector = other;
		},
		[min, max](std::vector<ivariable<T>*> args) {
			ivariable<T>* a = args.front();
			return clip_by_value(a->get_gradient(), min, max);
		}, 
	nnutils::formatter() << "clip_val(" << a->get_name() << ")");
	return op;
}

template<typename T>
ivariable<T>* operator + (T a, const ivariable<T>* b) {
	return new constant<T>(a) + b;
}

template<typename T>
ivariable<T>* operator + (const ivariable<T>* a, T b) {
	return a + new constant<T>(b);
}

template <typename T>
ivariable<T>* operator + (const ivariable<T>* a, const ivariable<T>* b) {
	if (nullptr == a) return b;
	else if (nullptr == b) return a;

	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other) { collector += other; },
		[](std::vector<ivariable<T>*> args) {
			// h'(f(x), g(x)) = f'(x) + g'(x)
			auto it = args.begin();
			ivariable<T>* res = (*it)->get_gradient();
			for (it++; args.end() != it; it++) {
				res = res + (*it)->get_gradient();
			}
			return res;
		}, 
	nnutils::formatter() << "(" << a->get_name() << "+" << b->get_name() << ")");
	return op;
}

template<typename T>
ivariable<T>* operator - (T a, const ivariable<T>* b) {
	return new constant<T>(a) - b;
}

template<typename T>
ivariable<T>* operator - (const ivariable<T>* a, T b) {
	return a - new constant<T>(b);
}

template <typename T>
ivariable<T>* operator - (const ivariable<T>* a, const ivariable<T>* b) {
	if (nullptr == a) return b;
	else if (nullptr == b) return a;

	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other) { collector -= other; },
		[](std::vector<ivariable<T>*> args) {
			// h'(f(x), g(x)) = f'(x) - g'(x)
			auto it = args.begin();
			ivariable<T>* res = (*it)->get_gradient();
			for (it++; args.end() != it; it++) {
				res = res - (*it)->get_gradient();
			}
			return res;
		}, 
	nnutils::formatter() << "(" << a->get_name() << "-" << b->get_name() << ")");
	return op;
}

template<typename T>
ivariable<T>* operator * (T a, const ivariable<T>* b) {
	return new constant<T>(a) * b;
}

template<typename T>
ivariable<T>* operator * (const ivariable<T>* a, T b) {
	return a * new constant<T>(b);
}

template <typename T>
ivariable<T>* operator * (const ivariable<T>* a, const ivariable<T>* b) {
	if (nullptr == a || nullptr == b) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other) { collector *= other; },
		[](std::vector<ivariable<T>*> args) {
			// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
			ivariable<T>* a = args.front();
			ivariable<T>* b = args.back();
			return a->get_gradient() * b + b->get_gradient() * a;
		}, 
	nnutils::formatter() << a->get_name() << "*" << b->get_name());
	return op;
}

template<typename T>
ivariable<T>* operator / (T a, const ivariable<T>* b) {
	return new constant<T>(a) / b;
}

template<typename T>
ivariable<T>* operator / (const ivariable<T>* a, T b) {
	return a / new constant<T>(b);
}

template <typename T>
ivariable<T>* operator / (const ivariable<T>* a, const ivariable<T>* b) {
	if (nullptr == a) return nullptr;
	assert (nullptr != b); // don't allow infinity

	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a, b},
		[](T& collector, T other) { collector /= other; },
		[](std::vector<ivariable<T>*> args) {
			// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
			ivariable<T>* a = args.front();
			ivariable<T>* b = args.back();
			return (a->get_gradient() * b - b->get_gradient() * a) / (b * b);
		}, 
	nnutils::formatter() << a->get_name() << "/" << b->get_name());
	return op;
}

}

#endif
