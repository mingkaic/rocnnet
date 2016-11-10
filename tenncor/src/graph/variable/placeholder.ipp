//
//  placeholder.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef placeholder_hpp

namespace nnet {

// PLACEHOLDER IMPLEMENTATION

template <typename T>
placeholder<T>::placeholder (const tensor_shape& shape, std::string name) :
    ileaf(shape, new typename ileaf<T>::open_init(this->out_), name) {}

template <typename T>
placeholder<T>::placeholder (const tensor_shape& shape, 
    initializer<T>* init, 
    std::string name = "") : ileaf(shape, init, name) {
	this->out_.allocate(std::make_shared<memory_alloc>());
	this->init_(this->out_);
	this->is_init = true;
}
		    
template <typename T>
placeholder<T>::placeholder (const placeholder<T>& other, std::string name) {
	this->copy(other, name);
}

template <typename T>
EVOKER_PTR<T> placeholder<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new placeholder(*this, name));
}

// changes shape
template <typename T>
ivariable<T>& placeholder<T>::operator = (VAR_PTR<T> other) {
	if (this != other.get()) {
	    if (false == this->out_.is_alloc()) {
			this->out_.allocate(
				std::make_shared<memory_alloc>(),
				other->get_shape());
		}
		this->out_ = this->get_tensor_from(other);
	}

	this->is_init = other->is_init;
	this->notify();
	return *this;
}

// maintains shape
template <typename T>
ivariable<T>& placeholder<T>::operator = (std::vector<T> data) {
	// note: if this is allocated,
	// compatibility is compared to allocated shape instead of allowed
	assert(this->out_.is_compatible_with(data));

	if (false == this->out_.is_alloc()) {
		this->out_.allocate(
			std::make_shared<memory_alloc>(),
			this->out_.guess_shape(data));
	}
	typename ileaf<T>::open_init* assigner =
			dynamic_cast<typename ileaf<T>::open_init*>(this->init_);
	*assigner = data;

	this->is_init = true;
	this->notify();
	return *this;
}

// changes shape
template <typename T>
ivariable<T>& placeholder<T>::operator = (const tensor<T>& data) {
	assert(this->out_.is_compatible_with(data));

	this->out_ = data;
	
	this->is_init = true;
	this->notify();
	return *this;
}

}

#endif
