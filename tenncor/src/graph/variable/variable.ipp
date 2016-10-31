//
//  variable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef variable_hpp

namespace nnet {

// CONSTANT IMPLEMENTATION

template <typename T>
constant<T>::constant (T scalar) {
	this->_out.set_shape(std::vector<size_t>{1});
	this->_out.allocate(std::make_shared<memory_alloc>());
	const_init<T>* cinit;
	this->init = cinit = new const_init<T>(scalar);
	(*cinit)(this->_out);
	this->name = nnutils::formatter() << scalar;
	this->is_init = true;
}

template <typename T>
constant<T>::constant (std::vector<T> raw, tensor_shape shape) {
	this->_out.set_shape(shape);
	this->_out.allocate(std::make_shared<memory_alloc>());
	typename ivar_init<T>::open_init* oinit;
	this->init = oinit = new typename ivar_init<T>::open_init(this->_out);
	(*oinit)(this->_out);
	(*oinit) = raw;
	this->name = nnutils::formatter() << raw.front() << ".." << raw.end();
	this->is_init = true;
}

template <typename T>
constant<T>::constant (VAR_PTR<T> get_out) {
	this->_out = get_out->eval();
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
	return ivariable<T>::make_shared(new variable(*this, name), true);
}

template <typename T>
variable<T>::variable (T scalar) {
	this->_out.set_shape(std::vector<size_t>{1});
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
	this->_out.set_shape(shape);
}

template <typename T>
variable<T>::variable (const tensor_shape& shape, initializer<T>& init, std::string name)
	: variable(shape, name) {
	this->init = init.clone();
}

template <typename T>
tensor<T>& variable<T>::initialize (void) {
	assert(this->init != nullptr);
	if (false == this->_out.is_alloc()) { // if not alloc, allocate
		this->_out.allocate(std::make_shared<memory_alloc>());
	}
	(*(this->init))(this->_out);
	this->is_init = true;
	return this->_out;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensor_shape alloc_shape) {
	assert(this->init != nullptr);
	if (false == this->_out.is_alloc()) { // if not alloc, allocate
		this->_out.allocate(std::make_shared<memory_alloc>(), alloc_shape);
	}
	(*(this->init))(this->_out);
	this->is_init = true;
	return this->_out;
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
	this->init = new typename ivar_init<T>::open_init(this->_out);
}

template <typename T>
placeholder<T>::placeholder (const tensor_shape& shape, std::string name) {
	this->name = name;
	this->_out.set_shape(shape);
	this->init = new typename ivar_init<T>::open_init(this->_out);
}

template <typename T>
EVOKER_PTR<T> placeholder<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new placeholder(*this, name));
}

// changes shape
template <typename T>
ivariable<T>& placeholder<T>::assign (VAR_PTR<T> other) {
	if (this != other.get()) {
		bool reshape_needed = false == other->_out.is_same_size(this->_out);
		if (false == this->_out.is_alloc()) {
			this->_out.allocate(
				std::make_shared<memory_alloc>(),
				other->get_shape());
		}
		this->_out = other->_out;
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
	assert(this->_out.is_compatible_with(data));

	if (false == this->_out.is_alloc()) {
		this->_out.allocate(
			std::make_shared<memory_alloc>(),
			this->_out.guess_shape(data));
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
	assert(this->_out.is_compatible_with(data));

	bool reshape_needed =
		this->_out.get_shape().is_fully_defined() &&
		!data.is_same_size(this->_out);

	this->_out = data;
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
	if (false == other._out.is_same_size(this->_out)) {
		consumer_reshape();
	}
}

}

#endif
