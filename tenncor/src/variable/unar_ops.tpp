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
ivariable<T>& iunar_ops<T>::operator () (VAR_PTR<T> var) {
    std::stringstream ns;
    ns << "<" << get_symb() << ">(" << var->get_name() << ")";
    this->name = ns.str();
    this->consume(*(var.get()));
    this->var = var;
	if (session::pre_shape_eval()) {
		shape_eval();
	}
    return *this;
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
	return std::shared_ptr<expose<T> >(new expose(*this, name));
}

template <typename T>
const tensor<T>& expose<T>::eval (void) {
	this->out = this->var->eval();
	return this->out;
}

template <typename T>
std::vector<T> expose<T>::get_raw (void) {
	// pass out from protected accessor
	return this->get_vec(this->eval());
}

template <typename T>
std::vector<T> expose<T>::get_derive (WEAK_VAR_PTR<T> over) const {
	tensor<T>* tense = this->gradient(over);
	if (nullptr == tense) {
		return std::vector<T>();
	}
	std::vector<T> vec = this->get_vec(*tense);
	delete tense;
	return vec;
}

// GRADIENT NODE

template <typename T>
EVOKER_PTR<T> gradient<T>::clone_impl (std::string name) {
	return std::shared_ptr<gradient<T> >(new gradient(*this, name));
}

template <typename T>
const tensor<T>& gradient<T>::eval (void) {
	tensor<T>* prime = this->var->gradient(over);
	this->out = *prime;
	delete prime;
	return this->out;
}

// USED FOR ELEMENT WISE OPERATIONS ONLY

template <typename T>
const tensor<T>& iunar_elem_ops<T>::eval (void) {
	assert(nullptr != this->var);
	const tensor<T>& evar = this->var->eval();
	tensor<T>* eptr = this->util_op(evar, get_op());
	this->out = *eptr;
	delete eptr;
	return this->out;
}

// NEGATION

template <typename T>
std::function<T(T)> neg<T>::get_op (void) { return shared_cnnet::op_neg<T>; }

template <typename T>
tensor<T>* neg<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // neg'(f(x)) = -f'(x)
        tensor<T>* ans = this->util_op(*deriv, shared_cnnet::op_neg<T>);
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> neg<T>::clone_impl (std::string name) {
	return std::shared_ptr<neg<T> >(new neg(*this, name));
}

// SINE

template <typename T>
std::function<T(T)> sin<T>::get_op (void) { return shared_cnnet::op_sin<T>; }

template <typename T>
tensor<T>* sin<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // sin'(f(x)) = f'(x)*cos(f(x))
        tensor<T>* cosres = this->util_op(this->get_eval(this->var), shared_cnnet::op_cos<T>);
        tensor<T>* ans = this->util_op(*deriv, *cosres, shared_cnnet::op_mul<T>);
        delete cosres;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> sin<T>::clone_impl (std::string name) {
	return std::shared_ptr<sin<T> >(new sin(*this, name));
}

// COSINE

template <typename T>
std::function<T(T)> cos<T>::get_op (void) { return shared_cnnet::op_cos<T>; }

template <typename T>
tensor<T>* cos<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // cos'(f(x)) = -f'(x)*sin(f(x))
        tensor<T>* sinres = this->util_op(this->get_eval(this->var), shared_cnnet::op_sin<T>);
        tensor<T>* negres = this->util_op(*deriv, shared_cnnet::op_neg<T>);
        tensor<T>* ans = this->util_op(*negres, *sinres, shared_cnnet::op_mul<T>);
        delete sinres;
        delete negres;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> cos<T>::clone_impl (std::string name) {
	return std::shared_ptr<cos<T> >(new cos(*this, name));
}

// TANGENT

template <typename T>
std::function<T(T)> tan<T>::get_op (void) { return shared_cnnet::op_tan<T>; }

template <typename T>
tensor<T>* tan<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // sec'(f(x)) = f'(x)*sec^2(f(x))
        // better with = f'(x)/cos^2(f(x))
        tensor<T>* cosres = this->util_op(this->get_eval(this->var), shared_cnnet::op_cos<T>);
        tensor<T>* cossqr = this->util_op(*cosres, *cosres, shared_cnnet::op_mul<T>); // cos^2(f(x))
        tensor<T>* ans = this->util_op(*deriv, *cossqr, shared_cnnet::op_div<T>);
        delete cosres;
        delete cossqr;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> tan<T>::clone_impl (std::string name) {
	return std::shared_ptr<tan<T> >(new tan(*this, name));
}

// COSECANT

template <typename T>
std::function<T(T)> csc<T>::get_op (void) { return shared_cnnet::op_csc<T>; }

template <typename T>
tensor<T>* csc<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
        // better with -f'(x)/(sin(f(x)*tan(f(x))))
        tensor<T>& value = this->get_eval(this->var);
        tensor<T>* tanres = this->util_op(value, shared_cnnet::op_tan<T>);
        tensor<T>* sinres = this->util_op(value, shared_cnnet::op_sin<T>);
        tensor<T>* negres = this->util_op(*deriv, shared_cnnet::op_neg<T>);
        // sin(f(x)*tan(f(x)))
        tensor<T>* pres = this->util_op(*sinres, *tanres, shared_cnnet::op_mul<T>);
        tensor<T>* ans = this->util_op(*negres, *pres, shared_cnnet::op_div<T>);
        delete tanres;
        delete sinres;
        delete negres;
        delete pres;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> csc<T>::clone_impl (std::string name) {
	return std::shared_ptr<csc<T> >(new csc(*this, name));
}

// SECANT

template <typename T>
std::function<T(T)> sec<T>::get_op (void) { return shared_cnnet::op_sec<T>; }

template <typename T>
tensor<T>* sec<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
        // better with f'(x)*tan(f(x))/cos(f(x))
        tensor<T>& value = this->get_eval(this->var);
        tensor<T>* tanres = this->util_op(value, shared_cnnet::op_tan<T>);
        tensor<T>* cosres = this->util_op(value, shared_cnnet::op_cos<T>);
        // tan(f(x))/cos(f(x))
        tensor<T>* dres = this->util_op(*tanres, *cosres, shared_cnnet::op_div<T>);
        tensor<T>* ans = this->util_op(*deriv, *dres, shared_cnnet::op_mul<T>);
        delete tanres;
        delete cosres;
        delete dres;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> sec<T>::clone_impl (std::string name) {
	return std::shared_ptr<sec<T> >(new sec(*this, name));
}

// COTANGENT

template <typename T>
std::function<T(T)> cot<T>::get_op (void) { return shared_cnnet::op_cot<T>; }

template <typename T>
tensor<T>* cot<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // cot'(f(x)) = -f'(x)*csc^2(f(x))
        tensor<T>* cscres = this->util_op(this->get_eval(this->var), shared_cnnet::op_csc<T>);
        tensor<T>* cscsqr = this->util_op(*cscres, *cscres, shared_cnnet::op_mul<T>); // csc^2(f(x))
        tensor<T>* negf = this->util_op(*deriv, shared_cnnet::op_neg<T>); // -f'(x)
        tensor<T>* ans = this->util_op(*negf, *cscsqr, shared_cnnet::op_mul<T>);
        delete cscres;
        delete cscsqr;
        delete negf;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> cot<T>::clone_impl (std::string name) {
	return std::shared_ptr<cot<T> >(new cot(*this, name));
}

// EXPONENT OF E

template <typename T>
std::function<T(T)> exp<T>::get_op (void) { return shared_cnnet::op_exp<T>; }

template <typename T>
tensor<T>* exp<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
    tensor<T>* deriv = this->var->gradient(over);
    if (nullptr != deriv) {
        // exp'(f(x)) = f'(x)*exp(f(x))
        tensor<T>* expres = this->util_op(this->get_eval(this->var), shared_cnnet::op_exp<T>);
        tensor<T>* ans = this->util_op(*deriv, *expres, shared_cnnet::op_mul<T>);
        delete expres;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
EVOKER_PTR<T> exp<T>::clone_impl (std::string name) {
	return std::shared_ptr<exp<T> >(new exp(*this, name));
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
	return std::shared_ptr<clip_by_value<T> >(new clip_by_value(*this, name));
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
	return std::shared_ptr<clip_by_norm<T> >(new clip_by_norm(*this, name));
}

}

#endif
