//
//  assign.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef assign_hpp

namespace nnet {

template <typename T>
assign<T>::assign (variable<T>* dest, ivariable<T>* src, std::function<void(T&,T)> trans) :
	dest_(dest), transfer_(trans) { this->add(src); }

template <typename T>
void assign<T>::execute (void) {
	tensor<T>* out = dest_->get_eval();
	tensor<T>* in = this->srcs_[0]->get_eval();
	assert(out->is_same_size(*in));

	T* old_data = out->get_raw();
	const T* new_data = in->get_raw();

	size_t total = in->n_elems();
	for (size_t i = 0; i < total; i++) {
		transfer_(old_data[i], new_data[i]);
	}

	dest_->notify();
}

// assign sub

template <typename T>
assign_sub<T>::assign_sub (variable<T>* dest, ivariable<T>* src) :
	assign<T>(dest, src, [](T& target, T data) { target -= data; }) {}

}

#endif