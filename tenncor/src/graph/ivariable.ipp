//
//  ivariable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef ivariable_hpp

namespace nnet {

// INITIALIZERS

template <typename T>
void const_init<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		std::fill(raw_data, raw_data+size, value_);
	});
}

template <typename T>
void random_uniform<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		for (size_t i = 0; i < size; i++) {
			raw_data[i] =  distribution_(session::get_generator());
		}
	});
}

// VARIABLE INTERFACE

template <typename T>
void ivariable<T>::copy (
	ivariable<T> const & other,
	std::string name) {
	 out_ = other.out_;
	if (0 == name.size()) {
		name = other.name+"_cpy";
	}
	this->name = name;
}

template <typename T>
ivariable<T>::~ivariable (void){
	session& sess = session::get_instance();
	sess.unregister_obj(*this);
}

template <typename T>
ivariable<T>& ivariable<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

}

#endif