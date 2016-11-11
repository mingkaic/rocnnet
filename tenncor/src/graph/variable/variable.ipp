//
//  variable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef variable_hpp

namespace nnet {

// VARIABLE IMPLEMENTATION

template <typename T>
variable<T>::variable (const variable<T>& other, std::string name) {
	ileaf<T>::copy(other, name);
}

template <typename T>
ievoker<T>* variable<T>::clone_impl (std::string name) {
	return new variable(*this, name);
}

template <typename T>
variable<T>::variable (T scalar) :
	    ileaf<T>(std::vector<size_t>{1},
		new const_init<T>(scalar), 
		nnutils::formatter() << scalar) {
	initialize();
}

template <typename T>
variable<T>::variable (const tensor_shape& shape, std::string name) :
		variable(shape, nullptr, name) {}

template <typename T>
variable<T>::variable (const tensor_shape& shape, initializer<T>& init, std::string name) :
	    ileaf<T>(shape, init.clone(), name) {}

template <typename T>
tensor<T>& variable<T>::initialize (void) {
	assert(this->init_ != nullptr);
	if (false == this->out_.is_alloc()) { // if not alloc, allocate
		this->out_.allocate(std::make_shared<memory_alloc>());
	}
	(*this->init_)(this->out_);
	this->is_init = true;
	return this->out_;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensor_shape alloc_shape) {
	assert(this->init_ != nullptr);
	if (false == this->out_.is_alloc()) { // if not alloc, allocate
		this->out_.allocate(std::make_shared<memory_alloc>(), alloc_shape);
	}
	(*this->init_)(this->out_);
	this->is_init = true;
	return this->out_;
}

}

#endif
