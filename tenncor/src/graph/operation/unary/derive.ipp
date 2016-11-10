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
ievoker<T>* derive<T>::clone_impl (std::string name) {
	return new derive(*this, name);
}

template <typename T>
void derive<T>::update (void) {
    ioperation<T>* func = dynamic_cast<ioperation<T>*>(this->dependencies_[0]);
    assert(func);
	this->out_ = func->get_eval();
}

}

#endif