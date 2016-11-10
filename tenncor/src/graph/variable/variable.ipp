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
constant<T>::constant (T scalar) :
	    ileaf<T>(std::vector<size_t>{1},
		new const_init<T>(scalar), 
		nnutils::formatter() << scalar) {
	this->out_.allocate(std::make_shared<memory_alloc>());
	this->init_(this->out_);
	this->is_init = true;
}

template <typename T>
constant<T>::constant (std::vector<T> raw, tensor_shape shape) :
	    ileaf<T>(shape,
		new typename ileaf<T>::open_init(this->out_), 
		nnutils::formatter() << raw.front() << ".." << raw.back() << raw.end()) {
	this->out_.allocate(std::make_shared<memory_alloc>());
	(*this->init_) = raw;
	this->is_init = true;
}

template <typename T>
constant<T>::constant (const constant<T>& other, std::string name) {
	this->copy(other, name);
}

template <typename T>
ievoker<T>* constant<T>::clone_impl (std::string name) {
	return new constant(*this, name);
}

// VARIABLE IMPLEMENTATION

template <typename T>
variable<T>::variable (const variable<T>& other, std::string name) {
	this->copy(other, name);
}

template <typename T>
ievoker<T>* variable<T>::clone_impl (std::string name) {
	return new variable(*this, name);
}

template <typename T>
variable<T>::variable (T scalar) {
	this->out_.set_shape(std::vector<size_t>{1});
	this->init_ = new const_init<T>(scalar);
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
	this->out_.set_shape(shape);
}

template <typename T>
variable<T>::variable (const tensor_shape& shape, initializer<T>& init, std::string name)
	: variable(shape, name) {
	this->init_ = init.clone();
}

template <typename T>
tensor<T>& variable<T>::initialize (void) {
	assert(this->init_ != nullptr);
	if (false == this->out_.is_alloc()) { // if not alloc, allocate
		this->out_.allocate(std::make_shared<memory_alloc>());
	}
	(*(this->init_))(this->out_);
	this->is_init = true;
	return this->out_;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensor_shape alloc_shape) {
	assert(this->init_ != nullptr);
	if (false == this->out_.is_alloc()) { // if not alloc, allocate
		this->out_.allocate(std::make_shared<memory_alloc>(), alloc_shape);
	}
	(*(this->init_))(this->out_);
	this->is_init = true;
	return this->out_;
}

}

#endif
