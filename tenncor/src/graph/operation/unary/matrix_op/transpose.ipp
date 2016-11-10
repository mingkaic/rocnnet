//
//  transpose.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef transpose_hpp

namespace nnet {

// MATRIX TRANSPOSE

template <typename T>
void transpose<T>::setup_gradient (void) {
	std::vector<ivariable<T>*> args;
	for (subject* child : this->dependencies_) {
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
			this->grad = transpose<T>::make(arg->get_gradient());
		}
	}
}

template <typename T>
void transpose<T>::shape_eval (void) {
	if (this->var) {
		tensor_shape ts = this->var->get_shape();
		if (ts.is_fully_defined()) {
			this->update(this->transpose_shape(ts));
		}
	}
}

template <typename T>
transpose<T>::transpose (VAR_PTR<T> in) { this->init(in); }

template <typename T>
EVOKER_PTR<T> transpose<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new transpose(*this, name));
}

template <typename T>
const tensor<T>& transpose<T>::eval (void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	assert(nullptr != this->var);
	const tensor<T>& in = this->var->eval();
	tensor<T>* ans = this->transpose_op(in);
	this->out_ = *ans;
	delete ans;
	return this->out_;
}

}

#endif