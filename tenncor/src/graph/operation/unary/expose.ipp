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
EVOKER_PTR<T> expose<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new expose(*this, name));
}

template <typename T>
void expose<T>::update (void) {
	this->out_ = this->dependencies_->eval();
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

template <typename T>
std::vector<T> expose<T>::get_derive (VAR_PTR<T> over) const {
	return get_vec(this->var->calc_gradient(over));
}

}

#endif