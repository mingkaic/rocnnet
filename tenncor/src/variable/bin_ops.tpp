//
//  bin_ops.tpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef bin_ops_hpp

namespace nnet {

// BINARY OPERATIONS

template <typename T>
void ibin_ops<T>::copy (const ivariable<T>& other, std::string name) {
	if (const ibin_ops<T>* bptr = dynamic_cast<const ibin_ops<T>*>(&other)) {
		if (own) delete own;
		a = bptr->a;
		b = bptr->b;
		own = bptr->own;
	}
	ivariable<T>::copy(other, name);
}

template <typename T>
void ibin_ops<T>::shape_eval (void) {
	if (a->get_shape().is_fully_defined() &&
		b->get_shape().is_fully_defined()) {
		tensor_shape ts = this->get_element_shape(
			this->get_eval(*a),
			this->get_eval(*b));
		assert(ts.is_fully_defined()); // assert initial shape is at least valid (re-checked at eval time)
		this->update(ts);
	}
}

template <typename T>
ivariable<T>& ibin_ops<T>::operator () (ivariable<T>& a, ivariable<T>& b) {
	std::stringstream ns;
	ns << "(" << a.get_name() << get_symb() << b.get_name() << ")";
	this->name = ns.str();
	this->consume(a); this->consume(b);
	this->a = &a; this->b = &b;
	if (session::pre_shape_eval()) {
		shape_eval();
	}
	return *this;
}

template <typename T>
ivariable<T>& ibin_ops<T>::operator () (ivariable<T>& a, T b) {
	own = new scalar<T>(b); // need smart pointer
	return (*this)(a, *own);
}

template <typename T>
ivariable<T>& ibin_ops<T>::operator () (T a, ivariable<T>& b) {
	own = new scalar<T>(a);
	return (*this)(*own, b);
}

template <typename T>
ibin_ops<T>& ibin_ops<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
const tensor<T>& ibin_ops<T>::eval (void) {
	assert(nullptr != a && nullptr != b);
	const tensor<T>& at = a->eval();
	const tensor<T>& bt = b->eval();
	tensor<T>* eptr = this->util_op(at, bt, get_op());
	this->out = *eptr;
	delete eptr;
	return this->out;
}

// ADDITION

template <typename T>
std::function<T(T, T)> add<T>::get_op (void) { return shared_cnnet::op_add<T>; }

template <typename T>
tensor<T>* add<T>::calc_gradient (ivariable<T>* over) const {
	// h'(f(x), g(x)) = f'(x) + g'(x)
	tensor<T>* deriva = this->a->gradient(over);
	tensor<T>* derivb = this->b->gradient(over);
	if (deriva && derivb) {
		tensor<T>* ans = this->util_op(*deriva, *derivb, shared_cnnet::op_add<T>);
		delete deriva;
		delete derivb;
		deriva = ans;
	} else if (derivb) {
		deriva = derivb;
	}
	return deriva;
}

template <typename T>
add<T>* add<T>::clone (std::string name) {
	return new add<T>(*this, name);
}

// SUBTRACTION

template <typename T>
std::function<T(T, T)> sub<T>::get_op (void) { return shared_cnnet::op_sub<T>; }

template <typename T>
tensor<T>* sub<T>::calc_gradient (ivariable<T>* over) const {
	// h'(f(x), g(x)) = f'(x) - g'(x)
	tensor<T>* deriva = this->a->gradient(over);
	tensor<T>* derivb = this->b->gradient(over);
	if (deriva && derivb) {
		tensor<T>* ans = this->util_op(*deriva, *derivb, shared_cnnet::op_sub<T>);
		delete deriva;
		delete derivb;
		deriva = ans;
	} else if (derivb) {
		tensor<T>* ans = this->util_op(*derivb, shared_cnnet::op_neg<T>);
		delete derivb;
		deriva = ans;
	}
	return deriva;
}

template <typename T>
sub<T>* sub<T>::clone (std::string name) {
	return new sub<T>(*this, name);
}

// MULTIPLICATION

template <typename T>
std::function<T(T, T)> mul<T>::get_op (void) { return shared_cnnet::op_mul<T>; }

template <typename T>
tensor<T>* mul<T>::calc_gradient (ivariable<T>* over) const {
	// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
	tensor<T>* deriva = this->a->gradient(over);
	tensor<T>* derivb = this->b->gradient(over);
	tensor<T>* ans = nullptr;
	if (deriva && derivb) {
		tensor<T>& tensa = this->get_eval(*this->a);
		tensor<T>& tensb = this->get_eval(*this->b);
		tensor<T>* ta = this->util_op(*deriva, tensb, shared_cnnet::op_mul<T>); // f'(x)*g(x)
		tensor<T>* tb = this->util_op(*derivb, tensa, shared_cnnet::op_mul<T>); // f(x)*g'(x)
		ans = this->util_op(*ta, *tb, shared_cnnet::op_add<T>);
		delete ta;
		delete tb;
		delete deriva;
		delete derivb;
	} else if (deriva){
		// deriva -> derivb * evala
		ans = this->util_op(*deriva, this->get_eval(*this->b), shared_cnnet::op_mul<T>); // f'(x)*g(x)
		delete deriva;
	} else if (derivb) {
		// derivb -> deriva * evalb
		ans = this->util_op(this->get_eval(*this->a), *derivb, shared_cnnet::op_mul<T>); // f(x)*g'(x)
		delete derivb;
	}
	return ans;
}

template <typename T>
mul<T>* mul<T>::clone (std::string name) {
	return new mul<T>(*this, name);
}

// DIVISION

template <typename T>
std::function<T(T, T)> div<T>::get_op (void) { return shared_cnnet::op_div<T>; }

template <typename T>
tensor<T>* div<T>::calc_gradient (ivariable<T>* over) const {
	// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
	tensor<T>* deriva = this->a->gradient(over);
	tensor<T>* derivb = this->b->gradient(over);
	tensor<T>* ans = nullptr;
	// deriva or derivb can be null.
	if (deriva && derivb) {
		tensor<T>& tensa = this->get_eval(*this->a);
		tensor<T>& tensb = this->get_eval(*this->b);
		// use util operations to handle scalar cases
		tensor<T>* numer0 = this->util_op(*deriva, tensb, shared_cnnet::op_mul<T>); // f'(x)*g(x)
		tensor<T>* numer1 = this->util_op(*derivb, tensa, shared_cnnet::op_mul<T>); // f(x)*g'(x)
		tensor<T>* numer = this->util_op(*numer0, *numer1, shared_cnnet::op_sub<T>); // numerator
		tensor<T>* denom = this->util_op(tensb, tensb, shared_cnnet::op_mul<T>); // denominator
		ans = this->util_op(*numer, *denom, shared_cnnet::op_div<T>);
		delete numer0;
		delete numer1;
		delete numer;
		delete denom;
		delete deriva;
		delete derivb;
	} else if (deriva) {
		// derivb is null, so h'(f(x), g(x)) = f'(x)*g(x)/g^2(x) = f'(x)/g(x)
		ans = this->util_op(*deriva, this->get_eval(*this->b), shared_cnnet::op_div<T>);
		delete deriva;
	} else if (derivb) {
		// deriva is null, so h'(f(x), g(x)) = -f(x)*g'(x)/g^2(x)
		tensor<T>& tensa = this->get_eval(*this->a);
		tensor<T>& tensb = this->get_eval(*this->b);
		// use util operations to handle scalar cases
		tensor<T>* negres = this->util_op(*derivb, shared_cnnet::op_neg<T>);
		tensor<T>* numer = this->util_op(*negres, tensa, shared_cnnet::op_mul<T>); // -f(x)*g'(x)
		tensor<T>* denom = this->util_op(tensb, tensb, shared_cnnet::op_mul<T>); // g(x)*g(x)
		ans = this->util_op(*numer, *denom, shared_cnnet::op_div<T>);
		delete negres;
		delete numer;
		delete denom;
		delete derivb;
	}
	return ans;
}

template <typename T>
div<T>* div<T>::clone (std::string name) {
	return new div<T>(*this, name);
}

}

#endif
