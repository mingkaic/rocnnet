//
//  derive.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef derive_hpp

namespace nnet {

// GRADIENT NODE

template <typename T>
EVOKER_PTR<T> derive<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new derive(*this, name));;
}

template <typename T>
const tensor<T>& derive<T>::eval (void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	this->out_ = this->var->calc_gradient(over_);
	return this->out_;
}

}

#endif