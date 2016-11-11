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
	for (ccoms::subject* child : this->dependencies_) {
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
			this->grad_ = new transpose<T>(arg->get_gradient());
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
transpose<T>::transpose (ivariable<T>* in) : iunar_ops(in) {}

template <typename T>
ievoker<T>* transpose<T>::clone_impl (std::string name) {
	return new transpose(*this, name);
}

template <typename T>
void transpose<T>::update (void) {
    ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	assert(arg);
	const tensor<T>& in = arg->get_eval();
	tensor<T>* ans = this->transpose_op(in);
	this->out_ = *ans;
	delete ans;
}

}

#endif