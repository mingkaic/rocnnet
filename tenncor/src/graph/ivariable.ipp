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
	this->delegate_task(in, [this](T* raw, size_t len) {
		std::fill(raw, raw+len, value_);
	});
}

template <typename T>
void random_uniform<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw, size_t len) {
		for (size_t i = 0; i < len; i++) {
			raw[i] =  distribution_(session::get_generator());
		}
	});
}

// VARIABLE INTERFACE

template <typename T>
ivariable<T>::~ivariable (void){
	session& sess = session::get_instance();
	sess.unregister_obj(*this);
}

template <typename T>
void ivariable<T>::copy (const ivariable<T>& other, std::string name) {
	out_ = std::unique_ptr<tensor<T> >(other.out_->clone());
	if (0 == name.size()) {
		name_ = other.name_+"_cpy";
	} else {
		name_ = name;
	}
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