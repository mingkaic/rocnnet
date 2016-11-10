//
//  iunar_ops.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef unar_ops_hpp

namespace nnet {

// UNARY OPERATIONS

template<typename T>
void iunar_ops<T>::shape_eval(void) {
    if (ivariable<T>* var = dynamic_cast<ivariable<T>*>(this->dependencies_[0])) {
        tensor_shape ts = var->get_shape();
        if (ts.is_fully_defined()) {
            this->update(ts);
        }
	}
}

// USED FOR ELEMENT WISE OPERATIONS ONLY

template<typename T>
void iunar_elem_ops<T>::update(void) {
	ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	assert(nullptr != arg);
	const tensor<T> &evar = arg->eval();
	tensor<T> *eptr = this->util_op(evar, get_op());
	this->out_ = *eptr;
	delete eptr;
}

}

#endif
