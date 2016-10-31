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
void iunar_ops<T>::copy(const ivariable <T> &other, std::string name) {
	if (const iunar_ops<T> *uptr = dynamic_cast<const iunar_ops<T> *>(&other)) {
		var = uptr->var;
	}
	ivariable<T>::copy(other, name);
}

template<typename T>
void iunar_ops<T>::replace(ivariable <T> *food, VAR_PTR <T> newfood) {
	if (var.get() == food) var = newfood;
}

template<typename T>
void iunar_ops<T>::shape_eval(void) {
	tensor_shape ts = var->get_shape();
	if (ts.is_fully_defined()) {
		this->update(ts);
	}
}

template<typename T>
void iunar_ops<T>::init(VAR_PTR <T> var) {
	std::stringstream ns;
	ns << "<" << get_symb() << ">(" << var->get_name() << ")";
	this->name = ns.str();
	this->consume(*(var.get()));
	this->var = var;
	if (session::pre_shape_eval()) {
		shape_eval();
	}
}

template<typename T>
iunar_ops <T> &iunar_ops<T>::operator=(const ivariable <T> &other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

// USED FOR ELEMENT WISE OPERATIONS ONLY

template<typename T>
const tensor <T> &iunar_elem_ops<T>::eval(void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	assert(nullptr != this->var);
	const tensor<T> &evar = this->var->eval();
	tensor<T> *eptr = this->util_op(evar, get_op());
	this->_out = *eptr;
	delete eptr;
	return this->_out;
}

}

#endif
