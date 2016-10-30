//
//  variable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef variable_hpp
#include <iostream>

namespace nnet {

// initializers

template <typename T>
void const_init<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		std::fill(raw_data, raw_data+size, value);
	});
}

template <typename T>
void random_uniform<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		for (size_t i = 0; i < size; i++) {
			raw_data[i] = distribution(session::get_generator());
		}
	});
}

// variable interface

template <typename T>
void ivariable<T>::copy (
	ivariable<T> const & other,
	std::string name) {
	out = other.out;
	if (0 == name.size()) {
		name = other.name+"_cpy";
	}
	this->name = name;
}

template <typename T>
ivariable<T>::ivariable (void) {
	session& sess = session::get_instance();
	sess.register_obj(*this);
}

template <typename T>
ivariable<T>::~ivariable (void){
	session& sess = session::get_instance();
	sess.unregister_obj(*this);
	std::unordered_set<ioperation<T>*> copy = consumers;
	for (ioperation<T>* cons : copy) {
		cons->deconsume(*this);
	}
}

template <typename T>
ivariable<T>& ivariable<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

// variable implementation

template <typename T>
struct ivar_init<T>::open_init : public initializer<T> {
	private:
		tensor<T>* hold = nullptr;

	public:
		open_init (tensor<T>& in) : hold(&in) {}

		virtual void operator () (tensor<T>& in) {
			hold = &in;
		}
		virtual initializer<T>* clone (void) {
			return new open_init(*hold);
		}

		virtual ivar_init<T>::open_init& operator = (const std::vector<T>& in) {
			this->delegate_task(*hold, [&in](T* raw_data, size_t size) {
				std::copy(in.begin(), in.end(), raw_data);
			});
			return *this;
		}
};

template <typename T>
class ivariable<T>::gradient_leaf : public ivar_init<T> {
	private:
		gradient_leaf (WEAK_VAR_PTR<T> integral) {
			this->out.set_shape(std::vector<size_t>{1});
			this->out.allocate(std::make_shared<memory_alloc>());
			this->is_init = true;

			this->name = "leaf<" + integral.lock()->get_name() + ">";
			this->integral = integral;
			typename ivar_init<T>::open_init* assigner;
			this->init = assigner = new typename ivar_init<T>::open_init(this->out);
			*assigner = std::vector<T>{0}; // initialize as zero
		}

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name) {
			return gradient_leaf::make(this->integral);
		}

	public:
		static std::shared_ptr<gradient_leaf> make (WEAK_VAR_PTR<T> integral) {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new gradient_leaf(integral));
			return std::static_pointer_cast<gradient_leaf>(inst);
		}
		virtual ~gradient_leaf (void) {}

		virtual void activate (VAR_PTR<T> active) {
			if (false == this->out.is_alloc()) {
				this->out.allocate(std::make_shared<memory_alloc>(), std::vector<size_t>{1});
			}
			// perform matrix calculus when necessary
			// when tensor variables can encasulate other variables
			// e.g.: A = [x y] where dA/dx is possible
			typename ivar_init<T>::open_init* assigner =
				dynamic_cast<typename ivar_init<T>::open_init*>(this->init);
			if (active == this->integral.lock()) {
				*assigner = std::vector<T>{1};
			}
		}

		virtual void deactivate (void) {
			if (false == this->out.is_alloc()) {
				this->out.allocate(std::make_shared<memory_alloc>(), std::vector<size_t>{1});
			}
			// when tensor variables can encasulate other variables
			// e.g.: A = [x y] where dA/dx is possible
			typename ivar_init<T>::open_init* assigner =
					dynamic_cast<typename ivar_init<T>::open_init*>(this->init);
			*assigner = std::vector<T>{0};
		}
};

template <typename T>
ivar_init<T>& ivar_init<T>::operator = (const VAR_PTR<T>& other) {
	if (this != other.get()) {
		if (nullptr != this->init) {
			delete this->init;
		}

		if (const std::shared_ptr<ivar_init<T> > vptr = std::dynamic_pointer_cast<ivar_init<T> >(other)) {
			this->copy(*vptr);
		} else {
			ivariable<T>::copy(*other);
		}
	}
	return *this;
}

// CONSTANT IMPLEMENTATION

template <typename T>
constant<T>::constant (T scalar) {
	this->out.set_shape(std::vector<size_t>{1});
	this->out.allocate(std::make_shared<memory_alloc>());
	const_init<T>* cinit;
	this->init = cinit = new const_init<T>(scalar);
	(*cinit)(this->out);
	this->name = nnutils::formatter() << scalar;
	this->is_init = true;
}

template <typename T>
constant<T>::constant (std::vector<T> raw, tensor_shape shape) {
	this->out.set_shape(shape);
	this->out.allocate(std::make_shared<memory_alloc>());
	typename ivar_init<T>::open_init* oinit;
	this->init = oinit = new typename ivar_init<T>::open_init(this->out);
	(*oinit)(this->out);
	(*oinit) = raw;
	this->name = nnutils::formatter() << raw.front() << ".." << raw.end();
	this->is_init = true;
}

template <typename T>
constant<T>::constant (VAR_PTR<T> get_out) {
	this->out = get_out->eval();
	this->name = get_out->get_name();
	this->is_init = true;
}

template <typename T>
constant<T>::constant (const constant<T>& other, std::string name) {
	this->copy(other, name);
	this->is_init = true;
}

template <typename T>
EVOKER_PTR<T> constant<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new constant(*this, name));
}

// VARIABLE IMPLEMENTATION

template <typename T>
variable<T>::variable (const variable<T>& other, std::string name) {
	this->copy(other, name);
}

template <typename T>
EVOKER_PTR<T> variable<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new variable(*this, name));
}

template <typename T>
variable<T>::variable (T scalar) {
	this->out.set_shape(std::vector<size_t>{1});
	this->init = new const_init<T>(scalar);
	this->name = nnutils::formatter() << scalar;
	initialize();
}

template <typename T>
variable<T>::variable (std::string name) {
	this->name = name;
}

template <typename T>
variable<T>::variable (const tensor_shape& shape, std::string name) {
	this->name = name;
	this->out.set_shape(shape);
}

template <typename T>
variable<T>::variable (const tensor_shape& shape, initializer<T>& init, std::string name)
	: variable(shape, name) {
	this->init = init.clone();
}

template <typename T>
tensor<T>& variable<T>::initialize (void) {
	assert(this->init != nullptr);
	if (false == this->out.is_alloc()) { // if not alloc, allocate
		this->out.allocate(std::make_shared<memory_alloc>());
	}
	(*(this->init))(this->out);
	this->is_init = true;
	return this->out;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensor_shape alloc_shape) {
	assert(this->init != nullptr);
	if (false == this->out.is_alloc()) { // if not alloc, allocate
		this->out.allocate(std::make_shared<memory_alloc>(), alloc_shape);
	}
	(*(this->init))(this->out);
	this->is_init = true;
	return this->out;
}

// placeholder implementation

template <typename T>
void placeholder<T>::consumer_reshape (void) {
	for (ioperation<T>* cons : this->consumers) {
		cons->shape_eval();
	}
}

template <typename T>
placeholder<T>::placeholder (const placeholder<T>& other, std::string name) {
	this->copy(other, name);
}

template <typename T>
placeholder<T>::placeholder (std::string name) : variable<T>(name) {
	this->init = new typename ivar_init<T>::open_init(this->out);
}

template <typename T>
placeholder<T>::placeholder (const tensor_shape& shape, std::string name) {
	this->name = name;
	this->out.set_shape(shape);
	this->init = new typename ivar_init<T>::open_init(this->out);
}

template <typename T>
EVOKER_PTR<T> placeholder<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new placeholder(*this, name));
}

// changes shape
template <typename T>
ivariable<T>& placeholder<T>::assign (VAR_PTR<T> other) {
	if (this != other.get()) {
		bool reshape_needed = false == other->out.is_same_size(this->out);
		if (false == this->out.is_alloc()) {
			this->out.allocate(
				std::make_shared<memory_alloc>(),
				other->get_shape());
		}
		this->out = other->out;
		if (reshape_needed) {
			consumer_reshape();
		}
	}
	this->is_init = true;
	return *this;
}

// maintains shape
template <typename T>
ivariable<T>& placeholder<T>::operator = (std::vector<T> data) {
	// note: if this is allocated,
	// compatibility is compared to allocated shape instead of allowed
	assert(this->out.is_compatible_with(data));

	if (false == this->out.is_alloc()) {
		this->out.allocate(
			std::make_shared<memory_alloc>(),
			this->out.guess_shape(data));
	}
	typename ivar_init<T>::open_init* assigner =
			dynamic_cast<typename ivar_init<T>::open_init*>(this->init);
	*assigner = data;

	this->is_init = true;
	return *this;
}

// changes shape
template <typename T>
ivariable<T>& placeholder<T>::operator = (const tensor<T>& data) {
	assert(this->out.is_compatible_with(data));

	bool reshape_needed =
		this->out.get_shape().is_fully_defined() &&
		!data.is_same_size(this->out);

	this->out = data;
	if (reshape_needed) {
		consumer_reshape();
	}
	return *this;
}

// changes shape
template <typename T>
void placeholder<T>::replace (const placeholder<T>& other) {
	for (ioperation<T>* cons : this->consumers) {
		cons->replace(this, &other);
	}
	if (false == other.out.is_same_size(this->out)) {
		consumer_reshape();
	}
}

}

#endif
