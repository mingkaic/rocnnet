//
//  operation.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef operation_hpp

namespace nnet {

template <typename T>
static inline T identity (T in) { return in; }

template <typename T>
static inline T op_neg (T in) { return -in; }

// achieve numerical stability for trig functions
template <typename T>
static inline T op_sin (T in) { return std::sin(in); }

template <typename T>
static inline T op_cos (T in) { return std::cos(in); }

template <typename T>
static inline T op_tan (T in) { return std::tan(in); }

template <typename T>
static inline T op_csc (T in) { return 1/std::sin(in); }

template <typename T>
static inline T op_sec (T in) { return 1/std::cos(in); }

template <typename T>
static inline T op_cot (T in) { return std::cos(in)/std::sin(in); }

template <typename T>
static inline T op_exp (T in) { return std::exp(in); }

template <typename T>
static inline T op_add (T a, T b) { return a + b; }

template <typename T>
static inline T op_sub (T a, T b) { return a - b; }

template <typename T>
static inline T op_mul (T a, T b) { return a * b; }

template <typename T>
static inline T op_div (T a, T b) { return a / b; }

// OPERATION INTERFACE UTILITY FUNCTIONS

template <typename T>
tensor_shape ioperation<T>::get_element_shape (
    const tensor<T>& a,
    const tensor<T>& b) const {
    tensor_shape ts_out;
    if (1 == a.n_dims()) {
        ts_out = b.get_shape();
    } else if (1 == b.n_dims() || a.is_same_size(b)){
        ts_out = a.get_shape();
    }
    return ts_out;
}

template <typename T>
tensor_shape ioperation<T>::get_matrix_shape (
    const tensor<T>& t1, const tensor<T>& t2,
    bool transposeA, bool transposeB,
    size_t& common_dim) const {
    tensor_shape ts_out;
    tensor_shape t1s = t1.get_shape();
    tensor_shape t2s = t2.get_shape();
    if (2 >= t1s.n_dims() &&
        2 >= t2s.n_dims()) {
        std::vector<size_t> al = t1s.as_list();
        std::vector<size_t> bl = t2s.as_list();
        size_t ax = t1s.n_dims() ? al[0] : 0;
        size_t ay = t1s.n_dims() > 1 ? al[1] : 1;
        size_t bx = t2s.n_dims() ? bl[0] : 0;
        size_t by = t2s.n_dims() > 1 ? bl[1] : 1;
        size_t dimX, dimY, dimZ;
        if (transposeA && transposeB) {
            if (ay == bx) {
                common_dim = ay;
                ts_out = std::vector<size_t>{by, ax};
            }
        } else if (transposeA) {
            if (ay == by) {
                common_dim = ay;
                ts_out = std::vector<size_t>{bx, ax};
            }
        } else if (transposeB) {
            if (ax == bx) {
                common_dim = ax;
                ts_out = std::vector<size_t>{by, ay};
            }
        } else { // !(transposeA || transposeB)
            if (ax == by) {
                common_dim = ax;
                ts_out = std::vector<size_t>{bx, ay};
            }
        }
    }
    return ts_out;
}

template <typename T>
template <typename U>
void ioperation<T>::util_op (
    U& out, const tensor<T>& in,
    std::function<void(U&, T)> op) const {
    T* inraw = in.raw_data;
    for (size_t i = 0; i < in.n_elems(); i++) {
        op(out, inraw[i]);
    }
}

template <typename T>
tensor<T>* ioperation<T>::util_op (
    const tensor<T>& in,
    std::function<T(T)> op) const {
    memory_alloc alloc;
    tensor<T>* ans = new tensor<T>(in.get_shape());
    ans->allocate(alloc);
    T* inraw = in.raw_data;
    T* outraw = ans->raw_data;
    for (size_t i = 0; i < ans->n_elems(); i++) {
        outraw[i] = op(inraw[i]);
    }
    return ans;
}

template <typename T>
tensor<T>* ioperation<T>::util_op (
    const tensor<T>& a,
    const tensor<T>& b,
    std::function<T(T, T)> op) const {
    tensor_shape ts = get_element_shape(a, b);
    assert(0 < ts.n_dims());
    memory_alloc alloc;
    tensor<T>* ans;
    ans = new tensor<T>(ts);
    ans->allocate(alloc);

    if (1 == a.n_elems()) {
        T scalar = a.get({0});
        T* in = b.raw_data;
        T* out = ans->raw_data;
        for (size_t i = 0; i < ans->n_elems(); i++) {
            out[i] = op(scalar, in[i]);
        }
    } else if (1 == b.n_elems()) {
        T* in = a.raw_data;
        T scalar = b.get({0});
        T* out = ans->raw_data;
        for (size_t i = 0; i < ans->n_elems(); i++) {
            out[i] = op(in[i], scalar);
        }
    } else if (a.is_same_size(b)) {
        T* ina = a.raw_data;
        T* inb = b.raw_data;
        T* out = ans->raw_data;
        for (size_t i = 0; i < ans->n_elems(); i++) {
            out[i] = op(ina[i], inb[i]);
        }
    } else {
        throw std::invalid_argument(
            "cannot element-wise operate on tensors of vastly different shapes");
    }
    return ans;
}

template <typename T>
tensor<T>* ioperation<T>::transpose_op (
    tensor<T> const & in) const {
    // restrict shapes
    tensor_shape ins = in.get_shape();
    assert(ins.n_dims() == 2);
    std::vector<size_t> inl = ins.as_list();
    memory_alloc all;
    size_t dimX = inl[0];
    size_t dimY = inl[1];
    tensor<T>* ans = new tensor<T>(std::vector<size_t>({dimY, dimX}));
    ans->allocate(all);
    T* rawin = in.raw_data;
    T* rawout = ans->raw_data;
    // not in place so x = y+1 doesn't work
    for (size_t y = 0; y < dimY; y++) {
        for (size_t x = 0; x < dimX; x++) {
            rawout[y+x*dimY] = rawin[x+y*dimX];
        }
    }
    return ans;
}

// restrict matrices shape at evaluation time
template <typename T>
tensor<T>* ioperation<T>::matmul_op (
    tensor<T> const & a,
    tensor<T> const & b,
    bool transposeA,
    bool transposeB) const {
    size_t dimZ;
    tensor_shape ts = get_matrix_shape(a, b, transposeA, transposeB, dimZ);
    assert(ts.n_dims() > 0);
    std::vector<size_t> dims = ts.as_list();
    size_t dimX = dims[0];
    size_t dimY = dims[1];
    memory_alloc all;
    tensor<T>* ans = new tensor<T>(std::vector<size_t>({dimX, dimY}));
    ans->allocate(all);
    T* rawa = a.raw_data;
    T* rawb = b.raw_data;
    T* rawr = ans->raw_data;
    for (size_t y = 0; y < dimY; y++) {
        for (size_t x = 0; x < dimX; x++) {
            rawr[x+y*dimX] = 0;
            for (size_t z = 0; z < dimZ; z++) {
                size_t aidx = transposeA ? y+z*dimY : z+y*dimZ;
                size_t bidx = transposeB ? z+x*dimZ : x+z*dimX;
                rawr[x+y*dimX] += rawa[aidx] * rawb[bidx];
            }
        }
    }
    return ans;
}

template <typename T>
void ioperation<T>::update (tensor_shape candidate_shape) {
    // no point in propagating if the shape is undefined
    if (0 != candidate_shape.n_dims()) {
        this->out.set_shape(candidate_shape);
        // propagate to consumers
        for (ioperation<T>* consumer : this->consumers) {
            consumer->shape_eval();
        }
    }
}

// FUNCTION WRAPPER IMPLEMENTATION

template <typename T>
void univar_func<T>::clear (void) {
    for (ioperation<T>* ptrs : ownout) {
        delete ptrs;
    }
    fanin = fanout = nullptr;
}

template <typename T>
void univar_func<T>::copy (ivariable<T> const & other, std::string name) {
    if (const univar_func<T>* uptr = dynamic_cast<const univar_func<T>*>(&other)) {
        // shallow copy
        // no ownership
        fanout = uptr->fanout;
        fanin = uptr->fanin;
    }
    ivariable<T>::copy(other, name);
}

template <typename T>
univar_func<T>::univar_func (const univar_func& other, std::string name) {
    copy(other, name);
}

template <typename T>
void univar_func<T>::shape_eval (void) {
    tensor_shape ts = fanout->get_shape();
    if (ts.is_fully_defined()) {
        this->update(ts);
    }
}

template <typename T>
univar_func<T>::univar_func (
    std::function<void(ioperation<T>*&)> declare) {
    declare(fanout);
}

template <typename T>
univar_func<T>* univar_func<T>::clone (std::string name) {
    return new univar_func<T>(*this, name);
}

template <typename T>
ivariable<T>& univar_func<T>::operator () (ivariable<T>* input) {
    if (nullptr != input) {
        ioperation<T>* buffer;
        std::queue<ioperation<T>*> q;
        q.push(fanout);
        // connect input
        while (false == q.empty()) {
            buffer = q.front();
            q.pop();

            if (unar_ops<T>* uptr = dynamic_cast<unar_ops<T>*>(buffer)) {
                if (nullptr == uptr->var) {
                    (*uptr)(*input);
                } else if (ioperation<T>* inptr =
                    dynamic_cast<ioperation<T>*>(uptr->var)) {
                    q.push(inptr);
                }
            } else if (bin_ops<T>* bptr = dynamic_cast<bin_ops<T>*>(buffer)) {
                if (nullptr == bptr->a && nullptr == bptr->b) {
                    (*bptr)(*input, *input);
                } else {
                    if (ioperation<T>* ptr1 =
                        dynamic_cast<ioperation<T>*>(bptr->a)) {
                        q.push(ptr1);
                    }
                    if (ioperation<T>* ptr2 =
                        dynamic_cast<ioperation<T>*>(bptr->b)) {
                        q.push(ptr2);
                    }
                }
            }
        }
        fanin = input;
    }
    shape_eval();
    return *this;
}

template <typename T>
univar_func<T>& univar_func<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        clear();
        copy(other);
    }
    return *this;
}

template <typename T>
tensor<T>* univar_func<T>::derive (ivariable<T>* over) const {
    return fanout->derive(over);
}

template <typename T>
tensor<T>* univar_func<T>::derive (void) const {
    return fanout->derive(fanin);
}

template <typename T>
const tensor<T>& univar_func<T>::eval (void) {
    return fanout->eval();
}

// ACTIVATION FUNCTION

template <typename T>
sigmoid<T>::sigmoid (void) : univar_func<T>([this](ioperation<T>*& outop) {
    // f(x) = 1/(1+e^-x)
    ioperation<T>* negres = new neg<T>();
    ioperation<T>* expres = new exp<T>(*negres);
    ioperation<T>* denom = new add<T>(1, *expres);
    outop = new div<T>(1, *denom);
    this->ownout = { negres, expres, denom, outop };
}) {}

template <typename T>
tanh<T>::tanh (void) : univar_func<T>([this](ioperation<T>*& outop) {
    // f(x) = (e^(2*x)+1)/(e^(2*x)-1)
    ioperation<T>* pres = new add<T>(); // 2*x
    ioperation<T>* expres = new exp<T>(*pres);
    ioperation<T>* numer = new sub<T>(*expres, 1);
    ioperation<T>* denom = new add<T>(*expres, 1);
    outop = new div<T>(*numer, *denom);
    this->ownout = { pres, expres, numer, denom, outop };
}) {}

// SCALAR IMPLEMENTATION

template <typename T>
T scalar<T>::equivalent (T a, T b) {
	// bool cannot necessarly auto-cast to T,
	// but all T types are guaranteed to values 0 and 1
	return a == b ? 1 : 0;
}

template <typename T>
void scalar<T>::atleast1 (bool& reduce, T value) {
	reduce = reduce || value == 1;
}

template <typename T>
tensor<T>* scalar<T>::calc_derive (ivariable<T>* over) const {
	// derivative of scalar Y by any other variable X is
	// dY / dx for all elem x in X
	T selfval = this->out.get({0});
	tensor<T>& tense = this->get_eval(*over);
	tensor<T>* res = nullptr;
	// is a scalar
	if (scalar<T>* sptr = dynamic_cast<scalar<T>*>(over)) {
		// value matches
		if (selfval == tense.get({0})) {
			memory_alloc all;
			const_init<T> oneinit(1);
			res = new tensor<T>(over->get_shape());
			res->allocate(all);
			oneinit(*res);
		}
	} else {
		memory_alloc all;
		const_init<T> init(selfval);
		res = new tensor<T>(over->get_shape());
		res->allocate(all);
		init(*res);
		tensor<T>* ans = this->util_op(
			*res, this->out, equivalent);
		delete res;
		bool nonderiv = false;
		this->template util_op<bool>(nonderiv, *ans, atleast1);
		if (nonderiv) {
			res = ans;
		} else {
			res = nullptr;
			delete ans;
		}
	}
	return res;
}

template <typename T>
scalar<T>::scalar (scalar<T> const & other, std::string name) {
    ivariable<T>::copy(other, name);
}

template <typename T>
scalar<T>::scalar (T value) {
	memory_alloc all;
	const_init<T> init(value);
	this->out.set_shape(std::vector<size_t>({1}));
	this->out.allocate(all);
	init(this->out);
	std::stringstream namestream;
	namestream << value;
	this->name = namestream.str();
}

template <typename T>
scalar<T>* scalar<T>::clone (std::string name) {
    return new scalar<T>(*this, name);
}

// UNARY OPERATIONS

template <typename T>
void unar_ops<T>::copy (const ivariable<T>& other, std::string name) {
    if (const unar_ops<T>* uptr = dynamic_cast<const unar_ops<T>*>(&other)) {
        var = uptr->var;
    }
    ivariable<T>::copy(other, name);
}

template <typename T>
tensor<T>* unar_ops<T>::calc_derive (ivariable<T>* over) const {
    return this->var->derive(over);
}

template <typename T>
void unar_ops<T>::shape_eval (void) {
    tensor_shape ts = var->get_shape();
    if (ts.is_fully_defined()) {
        this->update(ts);
    }
}

template <typename T>
ivariable<T>& unar_ops<T>::operator () (ivariable<T>& var) {
    std::stringstream ns;
    ns << "<" << get_symb() << ">(" << var.get_name() << ")";
    this->name = ns.str();
    this->consume(var);
    this->var = &var;
    shape_eval();
    return *this;
}

template <typename T>
unar_ops<T>& unar_ops<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        copy(other);
    }
    return *this;
}

template <typename T>
const tensor<T>& unar_ops<T>::eval (void) {
    assert(nullptr != var);
    const tensor<T>& evar = var->eval();
    tensor<T>* eptr = this->util_op(evar, get_op());
    this->out = *eptr;
    delete eptr;
    return this->out;
}

// BINARY OPERATIONS

template <typename T>
void bin_ops<T>::copy (const ivariable<T>& other, std::string name) {
    if (const bin_ops<T>* bptr = dynamic_cast<const bin_ops<T>*>(&other)) {
        if (own) delete own;

        a = bptr->a;
        b = bptr->b;
        own = bptr->own;
    }
    ivariable<T>::copy(other, name);
}

template <typename T>
void bin_ops<T>::shape_eval (void) {
    if (a->get_shape().is_fully_defined() &&
        b->get_shape().is_fully_defined()) {
        this->update(
            this->get_element_shape(
                this->get_eval(*a),
                this->get_eval(*b)));
    }
}

template <typename T>
ivariable<T>& bin_ops<T>::operator () (ivariable<T>& a, ivariable<T>& b) {
    std::stringstream ns;
    ns << a.get_name() << get_symb() << b.get_name();
    this->name = ns.str();
    this->consume(a); this->consume(b);
    this->a = &a; this->b = &b;
    shape_eval();
    return *this;
}

template <typename T>
ivariable<T>& bin_ops<T>::operator () (ivariable<T>& a, T b) {
    std::stringstream ns;
    ns << a.get_name() << get_symb() << b;
    this->name = ns.str();
    this->consume(a);
    this->a = &a;
    own = this->b = new scalar<T>(b); // need smart pointer
    shape_eval();
    return *this;
}

template <typename T>
ivariable<T>& bin_ops<T>::operator () (T a, ivariable<T>& b) {
    std::stringstream ns;
    ns << a << get_symb() << b.get_name();
    this->name = ns.str();
    this->consume(b);
    own = this->a = new scalar<T>(a);
    this->b = &b;
    shape_eval();
    return *this;
}

template <typename T>
bin_ops<T>& bin_ops<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        copy(other);
    }
    return *this;
}

template <typename T>
const tensor<T>& bin_ops<T>::eval (void) {
    assert(nullptr != a && nullptr != b);
    const tensor<T>& at = a->eval();
    const tensor<T>& bt = b->eval();
    tensor<T>* eptr = this->util_op(at, bt, get_op());
    this->out = *eptr;
    delete eptr;
    return this->out;
}

// MATRIX TRANSPOSE

template <typename T>
tensor<T>* transpose<T>::calc_derive (ivariable<T>* over) const {
    return nullptr;
}

template <typename T>
void transpose<T>::shape_eval (void) {
    tensor_shape ts = var->get_shape();
    if (ts.is_fully_defined()) {
        this->update(ts);
    }
}

template <typename T>
transpose<T>::transpose (transpose<T> const & other, std::string name) {
    this->var = other.var;
    ivariable<T>::copy(other, name);
}

template <typename T>
transpose<T>::transpose (ivariable<T>& in) {
    (*this)(in);
}

template <typename T>
transpose<T>* transpose<T>::clone (std::string name) {
    return new transpose<T>(*this, name);
}

template <typename T>
ivariable<T>& transpose<T>::operator () (ivariable<T>& in) {
    std::stringstream ns;
    ns << "<tranpose>(" << in.get_name() << ")";
    this->name = ns.str();
    this->consume(in);
    this->var = &in;
    shape_eval();
    return *this;
}

template <typename T>
transpose<T>& transpose<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        if (const transpose<T>* mptr = dynamic_cast<const transpose<T>*>(&other)) {
            var = mptr->var;
        }
        ivariable<T>::copy(other);
    }
    return *this;
}

template <typename T>
const tensor<T>& transpose<T>::eval (void) {
    assert(nullptr != var);
    const tensor<T>& in = var->eval();
    tensor<T>* ans = this->transpose_op(in);
    this->out = *ans;
    delete ans;
    return this->out;
}

// MATRIX MULTIPLICATION

template <typename T>
tensor<T>* matmul<T>::calc_derive (ivariable<T>* over) const {
    return nullptr;
}

template <typename T>
void matmul<T>::shape_eval (void) {
    if (a->get_shape().is_fully_defined() &&
        b->get_shape().is_fully_defined()) {
        size_t common_dim;
        this->update(
            this->get_matrix_shape(
                this->get_eval(*this->a),
                this->get_eval(*this->b),
                transposeA, transposeB, common_dim));
    }
}

template <typename T>
matmul<T>::matmul (matmul<T> const & other, std::string name) {
    a = other.a;
    b = other.b;
    transposeA = other.transposeA;
    transposeB = other.transposeB;
    ivariable<T>::copy(other, name);
}

template <typename T>
matmul<T>::matmul (
    ivariable<T>& a,
    ivariable<T>& b,
    bool transposeA,
    bool transposeB) {
    (*this)(a, b, transposeA, transposeB);
}

template <typename T>
matmul<T>* matmul<T>::clone (std::string name) {
    return new matmul<T>(*this, name);
}

template <typename T>
ivariable<T>& matmul<T>::operator () (
    ivariable<T>& a,
    ivariable<T>& b,
    bool transposeA,
    bool transposeB) {
    std::stringstream ns;
    ns << a.get_name() << "X" << b.get_name();
    this->name = ns.str();
    this->consume(a);
    this->consume(b);
    this->a = &a;
    this->b = &b;
    this->transposeA = transposeA;
    this->transposeB = transposeB;
    size_t common_dim;
    shape_eval();
    return *this;
}

template <typename T>
matmul<T>& matmul<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        if (const matmul<T>* mptr = dynamic_cast<const matmul<T>*>(&other)) {
            a = mptr->a;
            b = mptr->b;
            transposeA = mptr->transposeA;
            transposeB = mptr->transposeB;
        }
        ivariable<T>::copy(other);
    }
    return *this;
}

template <typename T>
const tensor<T>& matmul<T>::eval (void) {
    assert(nullptr != a && nullptr != b);
    const tensor<T>& at = this->a->eval();
    const tensor<T>& bt = this->b->eval();
    tensor<T>* ans = this->matmul_op(at, bt, transposeA, transposeB);
    this->out = *ans;
    delete ans;
    return this->out;
}

// OUT NODE

template <typename T>
std::function<T(T)> expose<T>::get_op (void) { return identity<T>; }

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
std::function<T(T)> neg<T>::get_op (void) { return op_neg<T>; }

template <typename T>
tensor<T>* neg<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // neg'(f(x)) = -f'(x)
        tensor<T>* ans = this->util_op(*deriv, op_neg<T>);
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
std::function<T(T)> sin<T>::get_op (void) { return op_sin<T>; }

template <typename T>
tensor<T>* sin<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // sin'(f(x)) = f'(x)*cos(f(x))
        tensor<T>* cosres = this->util_op(this->get_eval(*this->var), op_cos<T>);
        tensor<T>* ans = this->util_op(*deriv, *cosres, op_mul<T>);
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
std::function<T(T)> cos<T>::get_op (void) { return op_cos<T>; }

template <typename T>
tensor<T>* cos<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // cos'(f(x)) = -f'(x)*sin(f(x))
        tensor<T>* sinres = this->util_op(this->get_eval(*this->var), op_sin<T>);
        tensor<T>* negres = this->util_op(*deriv, op_neg<T>);
        tensor<T>* ans = this->util_op(*negres, *sinres, op_mul<T>);
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
std::function<T(T)> tan<T>::get_op (void) { return op_tan<T>; }

template <typename T>
tensor<T>* tan<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // sec'(f(x)) = f'(x)*sec^2(f(x))
        // better with = f'(x)/cos^2(f(x))
        tensor<T>* cosres = this->util_op(this->get_eval(*this->var), op_cos<T>);
        tensor<T>* cossqr = this->util_op(*cosres, *cosres, op_mul<T>); // cos^2(f(x))
        tensor<T>* ans = this->util_op(*deriv, *cossqr, op_div<T>);
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
std::function<T(T)> csc<T>::get_op (void) { return op_csc<T>; }

template <typename T>
tensor<T>* csc<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
        // better with -f'(x)/(sin(f(x)*tan(f(x))))
        tensor<T>& value = this->get_eval(*this->var);
        tensor<T>* tanres = this->util_op(value, op_tan<T>);
        tensor<T>* sinres = this->util_op(value, op_sin<T>);
        tensor<T>* negres = this->util_op(*deriv, op_neg<T>);
        // sin(f(x)*tan(f(x)))
        tensor<T>* pres = this->util_op(*sinres, *tanres, op_mul<T>);
        tensor<T>* ans = this->util_op(*negres, *pres, op_div<T>);
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
std::function<T(T)> sec<T>::get_op (void) { return op_sec<T>; }

template <typename T>
tensor<T>* sec<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
        // better with f'(x)*tan(f(x))/cos(f(x))
        tensor<T>& value = this->get_eval(*this->var);
        tensor<T>* tanres = this->util_op(value, op_tan<T>);
        tensor<T>* cosres = this->util_op(value, op_cos<T>);
        // tan(f(x))/cos(f(x))
        tensor<T>* dres = this->util_op(*tanres, *cosres, op_div<T>);
        tensor<T>* ans = this->util_op(*deriv, *dres, op_mul<T>);
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
std::function<T(T)> cot<T>::get_op (void) { return op_cot<T>; }

template <typename T>
tensor<T>* cot<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // cot'(f(x)) = -f'(x)*csc^2(f(x))
        tensor<T>* cscres = this->util_op(this->get_eval(*this->var), op_csc<T>);
        tensor<T>* cscsqr = this->util_op(*cscres, *cscres, op_mul<T>); // csc^2(f(x))
        tensor<T>* negf = this->util_op(*deriv, op_neg<T>); // -f'(x)
        tensor<T>* ans = this->util_op(*negf, *cscsqr, op_mul<T>);
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
std::function<T(T)> exp<T>::get_op (void) { return op_exp<T>; }

template <typename T>
tensor<T>* exp<T>::calc_derive (ivariable<T>* over) const {
    tensor<T>* deriv = this->var->derive(over);
    if (nullptr != deriv) {
        // exp'(f(x)) = f'(x)*exp(f(x))
        tensor<T>* expres = this->util_op(this->get_eval(*this->var), op_exp<T>);
        tensor<T>* ans = this->util_op(*deriv, *expres, op_mul<T>);
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

// ADDITION

template <typename T>
std::function<T(T, T)> add<T>::get_op (void) { return op_add<T>; }

template <typename T>
tensor<T>* add<T>::calc_derive (ivariable<T>* over) const {
    // h'(f(x), g(x)) = f'(x) + g'(x)
    tensor<T>* deriva = this->a->derive(over);
    tensor<T>* derivb = this->b->derive(over);
    if (deriva && derivb) {
        tensor<T>* ans = this->util_op(*deriva, *derivb, op_add<T>);
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
std::function<T(T, T)> sub<T>::get_op (void) { return op_sub<T>; }

template <typename T>
tensor<T>* sub<T>::calc_derive (ivariable<T>* over) const {
    // h'(f(x), g(x)) = f'(x) - g'(x)
    tensor<T>* deriva = this->a->derive(over);
    tensor<T>* derivb = this->b->derive(over);
    if (deriva && derivb) {
        tensor<T>* ans = this->util_op(*deriva, *derivb, op_sub<T>);
        delete deriva;
        delete derivb;
        deriva = ans;
    } else if (derivb) {
        tensor<T>* ans = this->util_op(*derivb, op_neg<T>);
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
std::function<T(T, T)> mul<T>::get_op (void) { return op_mul<T>; }

template <typename T>
tensor<T>* mul<T>::calc_derive (ivariable<T>* over) const {
    // h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
    tensor<T>* deriva = this->a->derive(over);
    tensor<T>* derivb = this->b->derive(over);
    tensor<T>* ans = nullptr;
    if (deriva && derivb) {
        tensor<T>& tensa = this->get_eval(*this->a);
        tensor<T>& tensb = this->get_eval(*this->b);
        tensor<T>* ta = this->util_op(*deriva, tensb, op_mul<T>); // f'(x)*g(x)
        tensor<T>* tb = this->util_op(*derivb, tensa, op_mul<T>); // f(x)*g'(x)
        ans = this->util_op(*ta, *tb, op_add<T>);
        delete ta;
        delete tb;
        delete deriva;
        delete derivb;
    } else if (deriva){
        // deriva -> derivb * evala
        ans = this->util_op(*deriva, this->get_eval(*this->b), op_mul<T>); // f'(x)*g(x)
        delete deriva;
    } else if (derivb) {
        // derivb -> deriva * evalb
        ans = this->util_op(this->get_eval(*this->a), *derivb, op_mul<T>); // f(x)*g'(x)
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
std::function<T(T, T)> div<T>::get_op (void) { return op_div<T>; }

template <typename T>
tensor<T>* div<T>::calc_derive (ivariable<T>* over) const {
    // h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
    tensor<T>* deriva = this->a->derive(over);
    tensor<T>* derivb = this->b->derive(over);
    tensor<T>* ans = nullptr;
    // deriva or derivb can be null.
    if (deriva && derivb) {
        tensor<T>& tensa = this->get_eval(*this->a);
        tensor<T>& tensb = this->get_eval(*this->b);
        // use util operations to handle scalar cases
        tensor<T>* numer0 = this->util_op(*deriva, tensb, op_mul<T>); // f'(x)*g(x)
        tensor<T>* numer1 = this->util_op(*derivb, tensa, op_mul<T>); // f(x)*g'(x)
        tensor<T>* numer = this->util_op(*numer0, *numer1, op_sub<T>); // numerator
        tensor<T>* denom = this->util_op(tensb, tensb, op_mul<T>); // denominator
        ans = this->util_op(*numer, *denom, op_div<T>);
        delete numer0;
        delete numer1;
        delete numer;
        delete denom;
        delete deriva;
        delete derivb;
    } else if (deriva) {
        // derivb is null, so h'(f(x), g(x)) = f'(x)*g(x)/g^2(x) = f'(x)/g(x)
        ans = this->util_op(*deriva, this->get_eval(*this->b), op_div<T>);
        delete deriva;
    } else if (derivb) {
        // deriva is null, so h'(f(x), g(x)) = -f(x)*g'(x)/g^2(x)
        tensor<T>& tensa = this->get_eval(*this->a);
        tensor<T>& tensb = this->get_eval(*this->b);
        // use util operations to handle scalar cases
        tensor<T>* negres = this->util_op(*derivb, op_neg<T>);
        tensor<T>* numer = this->util_op(*negres, tensa, op_mul<T>); // -f(x)*g'(x)
        tensor<T>* denom = this->util_op(tensb, tensb, op_mul<T>); // g(x)*g(x)
        ans = this->util_op(*numer, *denom, op_div<T>);
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
