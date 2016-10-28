//
//  variable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef variable_hpp

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
void variable<T>::copy (
	variable<T> const & other,
	std::string name) {
	init = other.init->copy();
	ivariable<T>::copy(other, name);
}

template <typename T>
variable<T>::variable (const variable<T>& other, std::string name) {
	copy(other, name);
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
variable<T>::variable (
	tensor_shape const & shape,
	std::string name) {
	this->name = name;
	this->out.set_shape(shape);
}

template <typename T>
variable<T>::variable (tensor_shape const & shape, initializer<T>& init, std::string name)
	: variable(shape, name) {
	this->init = init.copy();
}

template <typename T>
EVOKER_PTR<T> variable<T>::clone_impl (std::string name) {
	return std::shared_ptr<variable<T> >(new variable<T>(*this, name));
}

template <typename T>
variable<T>::~variable (void) {
	if (nullptr != this->init) {
		delete this->init;
	}
}

template <typename T>
variable<T>& variable<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		if (nullptr != this->init) {
			delete this->init;
		}

		if (const variable<T>* vptr = dynamic_cast<const variable<T>*>(&other)) {
			copy(*vptr);
		} else {
			ivariable<T>::copy(other);
		}
	}
	return *this;
}

template <typename T>
tensor<T>& variable<T>::initialize (void) {
	assert(init != nullptr);
	if (false == this->out.is_alloc()) { // if not alloc, allocate
		this->out.allocate(std::make_shared<memory_alloc>());
	}
	(*init)(this->out);
	is_init = true;
	return this->out;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensor_shape alloc_shape) {
	assert(init != nullptr);
	if (false == this->out.is_alloc()) { // if not alloc, allocate
		this->out.allocate(std::make_shared<memory_alloc>(), alloc_shape);
	}
	(*init)(this->out);
	is_init = true;
	return this->out;
}

template <typename T>
const tensor<T>& variable<T>::eval (void) {
	assert(is_init);
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
struct placeholder<T>::open_init : public initializer<T> {
	std::vector<T> prime;

	virtual void operator () (tensor<T>& in) {
		this->delegate_task(in, [this](T* raw_data, size_t size) {
			std::copy(prime.begin(), prime.end(), raw_data);
		});
	}

	virtual initializer<T>* copy (void) { return new open_init(); }
};

template <typename T>
placeholder<T>::placeholder (std::string name) : variable<T>(name) {}

template <typename T>
placeholder<T>::placeholder (const tensor_shape& shape, std::string name)
	: variable<T>(shape, name) {}

template <typename T>
EVOKER_PTR<T> placeholder<T>::clone_impl (std::string name) {
	return std::shared_ptr<placeholder<T> >(new placeholder<T>(*this, name));
}

// changes shape
template <typename T>
variable<T>& placeholder<T>::assign (VAR_PTR<T> other) {
	if (VAR_PTR<T>(this) != other) {
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
variable<T>& placeholder<T>::operator = (std::vector<T> data) {
	// note: if this is allocated,
	// compatibility is compared to allocated shape instead of allowed
	assert(this->out.is_compatible_with(data));

	if (false == this->out.is_alloc()) {
		this->out.allocate(
			std::make_shared<memory_alloc>(),
			this->out.guess_shape(data));
	}
	this->init = new open_init();
	assert(data.size() <= this->out.n_elems());
	dynamic_cast<open_init*>(this->init)->prime = data;
	(*this->init)(this->out);

	delete this->init;
	this->init = nullptr;
	this->is_init = true;
	return *this;
}

// changes shape
template <typename T>
variable<T>& placeholder<T>::operator = (const tensor<T>& data) {
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
