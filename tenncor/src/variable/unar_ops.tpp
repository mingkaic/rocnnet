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
tensor<T>* iunar_ops<T>::calc_derive (ivariable<T>* over) const {
    return this->var->derive(over);
}

template <typename T>
void iunar_ops<T>::shape_eval (void) {
    tensor_shape ts = var->get_shape();
    if (ts.is_fully_defined()) {
        this->update(ts);
    }
}

template <typename T>
ivariable<T>& iunar_ops<T>::operator () (ivariable<T>& var) {
    std::stringstream ns;
    ns << "<" << get_symb() << ">(" << var.get_name() << ")";
    this->name = ns.str();
    this->consume(var);
    this->var = &var;
    shape_eval();
    return *this;
}

template <typename T>
iunar_ops<T>& iunar_ops<T>::operator = (const ivariable<T>& other) {
    if (this != &other) {
        copy(other);
    }
    return *this;
}

template <typename T>
const tensor<T>& iunar_ops<T>::eval (void) {
    assert(nullptr != var);
    const tensor<T>& evar = var->eval();
    tensor<T>* eptr = this->util_op(evar, get_op());
    this->out = *eptr;
    delete eptr;
    return this->out;
}

// OUT NODE

template <typename T>
std::function<T(T)> expose<T>::get_op (void) { return shared_cnnet::identity<T>; }

template <typename T>
expose<T>* expose<T>::clone (std::string name) {
    return new expose<T>(*this, name);
}

template <typename T>
std::vector<T> expose<T>::get_raw (void) {
    // pass out from protected accessor
    return this->get_vec(this->eval());
}

template <typename T>
std::vector<T> expose<T>::get_derive (ivariable<T>& over) const {
    tensor<T>* tense = this->derive(&over);
    if (nullptr == tense) {
        return std::vector<T>();
    }
    std::vector<T> vec = this->get_vec(*tense);
    delete tense;
    return vec;
}

// NEGATION

template <typename T>
std::function<T(T)> neg<T>::get_op (void) { return shared_cnnet::op_neg<T>; }

template <typename T>
tensor<T>* neg<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // neg'(f(x)) = -f'(x)
        tensor<T>* ans = this->util_op(*deriv, shared_cnnet::op_neg<T>);
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
neg<T>* neg<T>::clone (std::string name) {
    return new neg<T>(*this, name);
}

// SINE

template <typename T>
std::function<T(T)> sin<T>::get_op (void) { return shared_cnnet::op_sin<T>; }

template <typename T>
tensor<T>* sin<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // sin'(f(x)) = f'(x)*cos(f(x))
        tensor<T>* cosres = this->util_op(this->get_eval(*this->var), shared_cnnet::op_cos<T>);
        tensor<T>* ans = this->util_op(*deriv, *cosres, shared_cnnet::op_mul<T>);
        delete cosres;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
sin<T>* sin<T>::clone (std::string name) {
    return new sin<T>(*this, name);
}

// COSINE

template <typename T>
std::function<T(T)> cos<T>::get_op (void) { return shared_cnnet::op_cos<T>; }

template <typename T>
tensor<T>* cos<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // cos'(f(x)) = -f'(x)*sin(f(x))
        tensor<T>* sinres = this->util_op(this->get_eval(*this->var), shared_cnnet::op_sin<T>);
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
cos<T>* cos<T>::clone (std::string name) {
    return new cos<T>(*this, name);
}

// TANGENT

template <typename T>
std::function<T(T)> tan<T>::get_op (void) { return shared_cnnet::op_tan<T>; }

template <typename T>
tensor<T>* tan<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // sec'(f(x)) = f'(x)*sec^2(f(x))
        // better with = f'(x)/cos^2(f(x))
        tensor<T>* cosres = this->util_op(this->get_eval(*this->var), shared_cnnet::op_cos<T>);
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
tan<T>* tan<T>::clone (std::string name) {
    return new tan<T>(*this, name);
}

// COSECANT

template <typename T>
std::function<T(T)> csc<T>::get_op (void) { return shared_cnnet::op_csc<T>; }

template <typename T>
tensor<T>* csc<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
        // better with -f'(x)/(sin(f(x)*tan(f(x))))
        tensor<T>& value = this->get_eval(*this->var);
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
csc<T>* csc<T>::clone (std::string name) {
    return new csc<T>(*this, name);
}

// SECANT

template <typename T>
std::function<T(T)> sec<T>::get_op (void) { return shared_cnnet::op_sec<T>; }

template <typename T>
tensor<T>* sec<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
        // better with f'(x)*tan(f(x))/cos(f(x))
        tensor<T>& value = this->get_eval(*this->var);
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
sec<T>* sec<T>::clone (std::string name) {
    return new sec<T>(*this, name);
}

// COTANGENT

template <typename T>
std::function<T(T)> cot<T>::get_op (void) { return shared_cnnet::op_cot<T>; }

template <typename T>
tensor<T>* cot<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // cot'(f(x)) = -f'(x)*csc^2(f(x))
        tensor<T>* cscres = this->util_op(this->get_eval(*this->var), shared_cnnet::op_csc<T>);
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
cot<T>* cot<T>::clone (std::string name) {
    return new cot<T>(*this, name);
}

// EXPONENT OF E

template <typename T>
std::function<T(T)> exp<T>::get_op (void) { return shared_cnnet::op_exp<T>; }

template <typename T>
tensor<T>* exp<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // exp'(f(x)) = f'(x)*exp(f(x))
        tensor<T>* expres = this->util_op(this->get_eval(*this->var), shared_cnnet::op_exp<T>);
        tensor<T>* ans = this->util_op(*deriv, *expres, shared_cnnet::op_mul<T>);
        delete expres;
        delete deriv;
        deriv = ans;
    }
    return deriv;
}

template <typename T>
exp<T>* exp<T>::clone (std::string name) {
    return new exp<T>(*this, name);
}

}

#endif
