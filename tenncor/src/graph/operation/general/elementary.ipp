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
	this->grad_ = der_(args);
}

template <typename T>
ivariable<T>* elementary<T>::clone_impl (std::string name) {
	return new elementary(*this, name);
}

template <typename T>
tensorshape elementary<T>::shape_eval (void) {
	tensorshape first = std::vector<size_t>{1};
	for (ccoms::subject* sub : this->dependencies_) {
		if (ivariable<T>* v = dynamic_cast<ivariable<T>*>(sub)) {
			tensorshape s = v->get_shape();
			if (1 != first.n_dims() &&
			   1 != s.n_dims() &&
			   false == first.is_compatible_with(s)) {
				throw std::invalid_argument(
						"cannot element-wise operate on tensors of vastly different shapes");
			}
			if (s.n_dims() >= first.n_dims()) first = s;
		}
	}
	return first;
}

template <typename T>
elementary<T>::elementary (std::vector<ivariable<T>*> args,
	std::function<void(T&, T)> op, BUILD_DERIVE<T> der, std::string name) :
		ioperation<T>(args, name), for_each_(op), der_(der) {
	this->out_ = std::make_unique<tensor_op<T> >(
	[this](T*& dest, std::vector<const T*> srcs) {
		tensorshape ts = shape_eval();

		for (size_t i = 0; i < ts.n_elems(); i++) {
			auto it = srcs.begin();
			if (1 == srcs.size()) {
				dest[i] = 0;
			} else {
				// n-nary operator. init with first object
				dest[i] = (*it)[i];
				it++;
			}
			while (srcs.end() != it) {
				for_each_(dest[i], (*it)[i]);
				it++;
			}
		}
	}, new ram_alloc());
}

template <typename T>
void elementary<T>::update (ccoms::subject* caller) {
	tensor<T> one(1);
	std::vector<tensor<T>*> tens;
	this->valid_tensor = true;
	for (ccoms::subject* sub : this->dependencies_) {
		if (ivariable<T>* var = dynamic_cast<ivariable<T>*>(sub)) {
			tensor<T>* a = &one;
			if (caller != var) {
				a = var->get_eval();
				if (nullptr == a) {
					this->valid_tensor = false;
					break;
				}
			}
			tens.push_back(a);
		}
	}

	if (this->valid_tensor) {
		this->out_->set_shape(shape_eval());
		*(this->out_)(tens);
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
	"abs(" + a->get_name() + ")");
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
	"neg(" + a->get_name() + ")");
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
	"sin(" + a->get_name() + ")");
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
	"cos(" + a->get_name() + ")");
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
 	"tan(" + a->get_name() + ")");
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
	"csc(" + a->get_name() + ")");
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
	"sec(" + a->get_name() + ")");
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
	"cot(" + a->get_name() + ")");
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
	"exp(" + a->get_name() + ")");
	return op;
}

template <typename T>
ivariable<T>* clip_val (const ivariable<T>* a, T min, T max) {
	if (nullptr == a) return nullptr;
	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
		[min, max](T& collector, T other) {
			if (min > other) other = min;
			else if (max < other) other = max;
			collector = other;
		},
		[min, max](std::vector<ivariable<T>*> args) {
			ivariable<T>* a = args.front();
			return clip_val(a->get_gradient(), min, max);
		}, 
	"clip_val(" + a->get_name() + ")");
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
	"(" + a->get_name() + "+" + b->get_name() + ")");
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
	"(" + a->get_name() + "-" + b->get_name() + ")");
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
	"(" + a->get_name() + "*" + b->get_name() + ")");
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
	"(" + a->get_name() + "/" + b->get_name() + ")");
	return op;
}

}

#endif
