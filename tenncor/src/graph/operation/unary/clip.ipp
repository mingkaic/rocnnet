//
//  clip.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef clip_hpp

namespace nnet {

// CLIP BY NORM

template <typename T>
ievoker<T>* clip_by_norm<T>::clone_impl (std::string name) {
	return new clip_by_norm(*this, name);
}

template<typename T>
void clip_by_norm<T>::update(void) {
	T l2norm;
	this->template util_op<double>(l2norm, this->out_, [](T& out, T in) {
		out += sqrt(in);
	});
	l2norm = sqrt(l2norm);
	ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	assert(nullptr != arg);
	const tensor<T> &evar = arg->eval();
	tensor<T> *eptr = this->util_op(evar, [this, &l2norm](T in) {
		return in * cap / l2norm;
	});
	this->out_ = *eptr;
	delete eptr;
}

}

#endif