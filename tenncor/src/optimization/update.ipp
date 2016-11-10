//
//  update.tpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef update_hpp
#include <iostream>

namespace nnet {

template <typename T>
void update<T>::copy (update<T>& other) {
	this->dest_ = other.dest_;
	this->src_ = other.src_;
	this->assign_ = other.assign_;
}

template <typename T>
update<T>::update (update<T>& other) {
    copy(other);
}

template <typename T>
ievoker<T>* update<T>::clone_impl (std::string name) {
	return new update(*this);
}

template <typename T>
update<T>::update (variable<T>* dest, ivariable<T>* src) : dest_(dest), src_(src) {}

template <typename T>
update<T>::update (variable<T>* dest,
    ivariable<T>* src,
    std::function<void(T&,T)> assign) :
	    dest_(dest), src_(src), assign_(assign) {}

template <typename T>
const tensor<T>& update<T>::eval (void) {
	tensor<T>& out = dest_->grab_tensor();
	const tensor<T>& in = src_->eval();
	assert(out.is_same_size(in));

	T* old_data = this->get_raw(out);
	const T* new_data = this->get_raw(in);
	size_t total = in.n_elems();
	for (size_t i = 0; i < total; i++) {
		assign_(old_data[i], new_data[i]);
	}

	return out;
}

// assign sub

template <typename T>
update_sub<T>::update_sub (update_sub<T>& other) {
    copy(other);
}

template <typename T>
ievoker<T>* update_sub<T>::clone_impl (std::string name) {
	return new update_sub(*this);
}

template <typename T>
update_sub<T>::update_sub (variable<T>* dest, ivariable<T>* src) :
	update<T>(dest, src, [](T& d, T s) { d -= s; }) {}

}

#endif