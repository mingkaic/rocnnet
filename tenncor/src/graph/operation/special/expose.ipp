//
//  expose.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef expose_hpp

namespace nnet {

// OUT NODE

template<typename T>
tensorshape expose<T>::shape_eval (void) {
	tensorshape s = dynamic_cast<ivariable<T>*>(this->dependencies_[0])->get_shape();
	s.assert_is_fully_defined();
	return s;
}

template <typename T>
ivariable<T>* expose<T>::clone_impl (std::string name) {
	return new expose(*this, name);
}

template <typename T>
void expose<T>::update (ccoms::subject* caller) {
	ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	assert(arg);
	this->out_ = arg->get_eval();
	this->notify();
}

template <typename T>
std::vector<T> expose<T>::get_raw (void) {
	tensor<T>* ten = this->get_eval();
	T* raw = ten->get_raw();
	return std::vector<T>(raw, raw + ten->n_elems());
}

}

#endif