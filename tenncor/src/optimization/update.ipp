//
//  update.tpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef update_hpp

namespace nnet {

template <typename T>
update<T>::update (update<T>& other) {
	this->dest = other.dest;
	this->src = other.src;
	this->assign = other.assign;
}

template <typename T>
EVOKER_PTR<T> update<T>::clone_impl (std::string name) {
	return std::shared_ptr<update<T> >(new update(*this));
}

template <typename T>
update<T>::update (std::shared_ptr<variable<T> > dest, VAR_PTR<T> src) : dest(dest), src(src) {}

template <typename T>
update<T>::update (std::shared_ptr<variable<T> > dest, VAR_PTR<T> src,
					std::function<void(T&,T)> assign) :
					dest(dest), src(src), assign(assign) {}

template <typename T>
const tensor<T>& update<T>::eval (void) {
	tensor<T>& out = dest->_out;
	const tensor<T>& in = src->eval();
	assert(out.is_same_size(in));

	T* old_data = this->get_raw(out);
	const T* new_data = this->get_raw(in);
	size_t total = in.n_elems();
	for (size_t i = 0; i < total; i++) {
		assign(old_data[i], new_data[i]);
	}

	return out;
}

// assign sub

template <typename T>
update_sub<T>::update_sub (std::shared_ptr<variable<T> > dest, VAR_PTR<T> src) :
update<T>(dest, src, [](T& dest, T src) {
	dest -= src;
}) {}

}

#endif