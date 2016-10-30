//
//  iunar_ops.tpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef unar_ops_hpp

namespace nnet {

// UNARY OPERATIONS

template <typename T>
void iunar_ops<T>::copy (const ivariable<T>& other, std::string name) {
    if (const iunar_ops<T>* uptr = dynamic_cast<const iunar_ops<T>*>(&other)) {
        var = uptr->var;
    }
    ivariable<T>::copy(other, name);
}

template <typename T>
void iunar_ops<T>::replace (ivariable<T>* food, VAR_PTR<T> newfood) {
	if (var.get() == food) var = newfood;
}

template <typename T>
void iunar_ops<T>::shape_eval (void) {
	tensor_shape ts = var->get_shape();
	if (ts.is_fully_defined()) {
		this->update(ts);
	}
}

template <typename T>
void iunar_ops<T>::init (VAR_PTR<T> var) {
	std::stringstream ns;
	ns << "<" << get_symb() << ">(" << var->get_name() << ")";
	this->name = ns.str();
	this->consume(*(var.get()));
	this->var = var;
	if (session::pre_shape_eval()) {
		shape_eval();
	}
}

template <typename T>
iunar_ops<T>& iunar_ops<T>::operator = (const ivariable<T>& other) {
    if (this != &other) {
        copy(other);
    }
    return *this;
}

// OUT NODE

template <typename T>
EVOKER_PTR<T> expose<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new expose(*this, name));
}

template <typename T>
const tensor<T>& expose<T>::eval (void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	return this->var->eval();
}

template <typename T>
std::vector<T> expose<T>::get_raw (void) {
	// pass out from protected accessor
	return this->get_vec(this->eval());
}

template <typename T>
std::vector<T> expose<T>::get_derive (VAR_PTR<T> over) const {
	return this->get_vec(this->var->calc_gradient(over));
}

// GRADIENT NODE

template <typename T>
EVOKER_PTR<T> gradient<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new gradient(*this, name));;
}

template <typename T>
const tensor<T>& gradient<T>::eval (void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	this->out = this->var->calc_gradient(over);
	return this->out;
}

// USED FOR ELEMENT WISE OPERATIONS ONLY

template <typename T>
const tensor<T>& iunar_elem_ops<T>::eval (void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	assert(nullptr != this->var);
	const tensor<T>& evar = this->var->eval();
	tensor<T>* eptr = this->util_op(evar, get_op());
	this->out = *eptr;
	delete eptr;
	return this->out;
}

// CLIP ELEMENT VALUES

template <typename T>
std::function<T(T)> clip_by_value<T>::get_op (void) {
	return [this](T in) {
		if (min > in) return min;
		else if (max < in) return max;
		return in;
	};
}

template <typename T>
EVOKER_PTR<T> clip_by_value<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new clip_by_value(*this, name));
}

template <typename T>
std::function<T(T)> clip_by_norm<T>::get_op (void) {
	T l2norm;
	this->template util_op<double>(l2norm, this->out, [](T& out, T in) {
		out += sqrt(in);
	});
	l2norm = sqrt(l2norm);
	return [this, &l2norm](T in) {
		return in * cap / l2norm;
	};
}

template <typename T>
EVOKER_PTR<T> clip_by_norm<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new clip_by_norm(*this, name));
}

}

#endif
