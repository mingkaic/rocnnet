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

template <typename T>
ievoker<T>* expose<T>::clone_impl (std::string name) {
	return new expose(*this, name);
}

template <typename T>
void expose<T>::update (void) {
    ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	assert(arg);
	this->out_ = arg->get_eval();
	this->notify();
}

template <typename T>
std::vector<T> expose<T>::get_vec (const tensor<T>& in) const {
	const T* raw = ievoker<T>::get_raw(in);
	size_t limit = in.n_elems();
	return std::vector<T>(raw, raw + limit);
}

template <typename T>
std::vector<T> expose<T>::get_raw (void) {
	// pass out from protected accessor
	return get_vec(this->eval());
}

}

#endif