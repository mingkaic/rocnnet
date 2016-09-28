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
template <typename U>
void ioperation<T>::util_op (
    U& out,
    tensor<T> const & in,
    std::function<void(U&, T)> op) const {
    T* inraw = in.raw_data;
    for (size_t i = 0; i < in.n_elems(); i++) {
        op(out, inraw[i]);
    }
}

template <typename T>
tensor<T>* ioperation<T>::util_op (
    tensor<T> const & in,
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
    tensor<T> const & a,
    tensor<T> const & b,
    std::function<T(T, T)> op) const {
    memory_alloc alloc;
    tensor<T>* ans;
    if (1 == a.n_elems()) {
        ans = new tensor<T>(b.get_shape());
        ans->allocate(alloc);
        T scalar = a.get({0});
        T* in = b.raw_data;
        T* out = ans->raw_data;
        for (size_t i = 0; i < ans->n_elems(); i++) {
            out[i] = op(scalar, in[i]);
        }
    } else if (1 == b.n_elems()) {
        ans = new tensor<T>(a.get_shape());
        ans->allocate(alloc);
        T* in = a.raw_data;
        T scalar = b.get({0});
        T* out = ans->raw_data;
        for (size_t i = 0; i < ans->n_elems(); i++) {
            out[i] = op(in[i], scalar);
        }
    } else if (a.is_same_size(b)) {
        ans = new tensor<T>(a.get_shape());
        ans->allocate(alloc);
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
    // restrict shapes
    tensor_shape as = a.get_shape();
    tensor_shape bs = b.get_shape();
    assert(2 >= as.n_dims());
    assert(2 >= bs.n_dims());
    std::vector<size_t> al = as.as_list();
    std::vector<size_t> bl = bs.as_list();
    size_t ax = as.n_dims() ? al[0] : 0;
    size_t ay = as.n_dims() > 1 ? al[1] : 1;
    size_t bx = bs.n_dims() ? bl[0] : 0;
    size_t by = bs.n_dims() > 1 ? bl[1] : 1;
    size_t dimX, dimY, dimZ;
    if (transposeA && transposeB) {
        assert(ay == bx);
        dimZ = ay;
        dimX = by; dimY = ax;
    } else if (transposeA) {
        assert(ay == by);
        dimZ = ay;
        dimX = bx; dimY = ax;
    } else if (transposeB) {
        assert(ax == bx);
        dimZ = ax;
        dimX = by; dimY = ay;
    } else { // !(transposeA || transposeB)
        assert(ax == by);
        dimZ = ax;
        dimX = bx; dimY = ay;
    }
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
                size_t aidx = transposeA ? y+z*ax : z+y*ax;
                size_t bidx = transposeB ? z+x*bx : x+z*bx;
                rawr[x+y*dimX] += rawa[aidx] * rawb[bidx];
            }
        }
    }
    return ans;
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
void univar_func<T>::copy (
    univar_func<T> const & other,
    std::string name) {
    // shallow copy
    // no ownership
    fanout = other.fanout;
    fanin = other.fanin;
    ivariable<T>::copy(other, name);
}

template <typename T>
univar_func<T>::univar_func (univar_func<T> const & other, std::string name) {
    copy(other, name);
}

template <typename T>
univar_func<T>::univar_func (
    std::function<void(ioperation<T>*)> declare) {
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
    return *this;
}

template <typename T>
univar_func<T>& univar_func<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        clear();
        if (const univar_func<T>* uptr = dynamic_cast<const univar_func<T>*>(&other)) {
            copy(*uptr);
        } else {
            ivariable<T>::copy(other);
        }
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
sigmoid<T>::sigmoid (void) : univar_func<T>([this](ioperation<T>* outop) {
    // f(x) = 1/(1+e^-x)
    ioperation<T>* negres = new neg<T>();
    ioperation<T>* expres = new exp<T>(negres);
    ioperation<T>* denom = new add<T>(1, expres);
    outop = new div<T>(1, denom);
    this->ownout = { negres, expres, denom, outop };
}) {}

template <typename T>
tanh<T>::tanh (void) : univar_func<T>([this](ioperation<T>* outop) {
    // f(x) = (e^(2*x)+1)/(e^(2*x)-1)
    ioperation<T>* pres = new add<T>(); // 2*x
    ioperation<T>* expres = new exp<T>(pres);
    ioperation<T>* numer = new sub<T>(expres, 1);
    ioperation<T>* denom = new add<T>(expres, 1);
    outop = new div<T>(numer, denom);
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
void unar_ops<T>::init (std::string op, ivariable<T>& var) {
    std::stringstream ns;
    ns << "<" << op << ">(" << var.get_name() << ")";
    this->name = ns.str();
    this->consume(var);
    this->var = &var;
}

template <typename T>
unar_ops<T>::unar_ops (unar_ops<T> const & other, std::string name) {
    var = other.var;
    op = other.op;
    ivariable<T>::copy(other, name);
}

template <typename T>
tensor<T>* unar_ops<T>::calc_derive (ivariable<T>* over) const {
    return this->var->derive(over);
}

template <typename T>
 unar_ops<T>* unar_ops<T>::clone (std::string name) {
    return new unar_ops(*this, name);
}

template <typename T>
unar_ops<T>& unar_ops<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        if (const unar_ops<T>* uptr = dynamic_cast<const unar_ops<T>*>(&other)) {
            var = uptr->var;
            op = uptr->op;
        }
        ivariable<T>::copy(other);
    }
    return *this;
}

template <typename T>
const tensor<T>& unar_ops<T>::eval (void) {
    assert(nullptr != var);
    const tensor<T>& evar = var->eval();
    tensor<T>* eptr = this->util_op(evar, op);
    this->out = *eptr;
    delete eptr;
    return this->out;
}

// BINARY OPERATIONS

template <typename T>
bin_ops<T>::bin_ops (bin_ops<T> const & other, std::string name) {
    a = other.a;
    b = other.b;
    op = other.op;
    ivariable<T>::copy(other, name);
}

template <typename T>
bin_ops<T>* bin_ops<T>::clone (std::string name) {
    return new bin_ops(*this, name);
}

template <typename T>
void bin_ops<T>::init (std::string op, ivariable<T>& a, ivariable<T>& b) {
    std::stringstream ns;
    ns << a.get_name() << op << b.get_name();
    this->name = ns.str();
    this->consume(a); this->consume(b);
    this->a = &a; this->b = &b;
}

template <typename T>
void bin_ops<T>::init (std::string op, ivariable<T>& a, T b) {
    std::stringstream ns;
    ns << a.get_name() << op << b;
    this->name = ns.str();
    this->consume(a);
    this->a = &a; this->b = new scalar<T>(b); // need smart pointer
}

template <typename T>
void bin_ops<T>::init (std::string op, T a, ivariable<T>& b) {
    std::stringstream ns;
    ns << a << op << b.get_name();
    this->name = ns.str();
    this->consume(b);
    this->a = new scalar<T>(a); this->b = &b;
}

template <typename T>
bin_ops<T>& bin_ops<T>::operator = (ivariable<T> const & other) {
    if (this != &other) {
        if (const bin_ops<T>* bptr = dynamic_cast<const bin_ops<T>*>(&other)) {
            a = bptr->a;
            b = bptr->b;
            op = bptr->op;
        }
        ivariable<T>::copy(other);
    }
    return *this;
}

template <typename T>
const tensor<T>& bin_ops<T>::eval (void) {
    assert(nullptr != a && nullptr != b);
    const tensor<T>& at = a->eval();
    const tensor<T>& bt = b->eval();
    tensor<T>* eptr = this->util_op(at, bt, op);
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
ivariable<T>& expose<T>::operator () (ivariable<T>& in) {
    this->op = identity<T>;
    this->init("expose", in);
    return *this;
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
ivariable<T>& neg<T>::operator () (ivariable<T>& in) {
    this->op = op_neg<T>;
    this->init("-", in);
    return *this;
}

// SINE

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
ivariable<T>& sin<T>::operator () (ivariable<T>& in) {
    this->op = op_sin<T>;
    this->init("sin", in);
    return *this;
}

// COSINE

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
ivariable<T>& cos<T>::operator () (ivariable<T>& in) {
    this->op = op_cos<T>;
    this->init("cos", in);
    return *this;
}

// TANGENT

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
ivariable<T>& tan<T>::operator () (ivariable<T>& in) {
    this->op = op_tan<T>;
    this->init("tan", in);
    return *this;
}

// COSECANT

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
ivariable<T>& csc<T>::operator () (ivariable<T>& in) {
    this->op = op_csc<T>;
    this->init("csc", in);
    return *this;
}

// SECANT

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
ivariable<T>& sec<T>::operator () (ivariable<T>& in) {
    this->op = op_sec<T>;
    this->init("sec", in);
    return *this;
}

// COTANGENT

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
ivariable<T>& cot<T>::operator () (ivariable<T>& in) {
    this->op = op_cot<T>;
    this->init("cot", in);
    return *this;
}

// EXPONENT OF E

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
ivariable<T>& exp<T>::operator () (ivariable<T>& in) {
    this->op = op_exp<T>;
    this->init("exp", in);
    return *this;
}

// ADDITION

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
ivariable<T>& add<T>::operator () (ivariable<T>& a, ivariable<T>& b) {
    this->op = op_add<T>;
    this->init("+", a, b);
    return *this;
}

template <typename T>
ivariable<T>& add<T>::operator () (ivariable<T>& a, T b) {
    this->op = op_add<T>;
    this->init("+", a, b);
    return *this;
}

template <typename T>
ivariable<T>& add<T>::operator () (T a, ivariable<T>& b) {
    this->op = op_add<T>;
    this->init("+", a, b);
    return *this;
}

// SUBTRACTION

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
ivariable<T>& sub<T>::operator () (ivariable<T>& a, ivariable<T>& b) {
    this->op = op_sub<T>;
    this->init("-", a, b);
    return *this;
}

template <typename T>
ivariable<T>& sub<T>::operator () (ivariable<T>& a, T b) {
    this->op = op_sub<T>;
    this->init("-", a, b);
    return *this;
}

template <typename T>
ivariable<T>& sub<T>::operator () (T a, ivariable<T>& b) {
    this->op = op_sub<T>;
    this->init("-", a, b);
    return *this;
}

// MULTIPLICATION

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
ivariable<T>& mul<T>::operator () (ivariable<T>& a, ivariable<T>& b) {
    this->op = op_mul<T>;
    this->init("*", a, b);
    return *this;
}

template <typename T>
ivariable<T>& mul<T>::operator () (ivariable<T>& a, T b) {
    this->op = op_mul<T>;
    this->init("*", a, b);
    return *this;
}

template <typename T>
ivariable<T>& mul<T>::operator () (T a, ivariable<T>& b) {
    this->op = op_mul<T>;
    this->init("*", a, b);
    return *this;
}

// DIVISION

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
ivariable<T>& div<T>::operator () (ivariable<T>& a, ivariable<T>& b) {
    this->op = op_div<T>;
    this->init("/", a, b);
    return *this;
}

template <typename T>
ivariable<T>& div<T>::operator () (ivariable<T>& a, T b) {
    this->op = op_div<T>;
    this->init("/", a, b);
    return *this;
}

template <typename T>
ivariable<T>& div<T>::operator () (T a, ivariable<T>& b) {
    this->op = op_div<T>;
    this->init("/", a, b);
    return *this;
}

}

#endif
