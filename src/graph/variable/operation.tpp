//
//  operation.tpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef operation_hpp

namespace nnet {

// OPERATION INTERFACE UTILITY FUNCTIONS

template <typename T>
tensor_shape ioperation<T>::get_element_shape (
	const tensor<T>& a,
	const tensor<T>& b) const {
	tensor_shape ts_out;
	if (1 == a.n_elems()) {
		ts_out = b.get_shape();
	} else if (1 == b.n_elems() || a.is_same_size(b)){
		ts_out = a.get_shape();
	}
	return ts_out;
}

template <typename T>
tensor_shape ioperation<T>::transpose_shape (const tensor_shape& ins) const {
	// restrict shapes
	assert(ins.n_dims() == 2);
	std::vector<size_t> inl = ins.as_list();
	size_t dimX = inl[0];
	size_t dimY = inl[1];
	return tensor_shape(std::vector<size_t>{dimY, dimX});
}

template <typename T>
tensor_shape ioperation<T>::change_shape (
	const tensor_shape& ins,
	size_t index,
	double multiplier,
	size_t& below_dim,
	size_t& at_idx) const {
	ins.assert_is_fully_defined(); // can't change shape if shape info is not complete
	std::vector<size_t> tv = ins.as_list();
	// allocated additional space along index
	below_dim = 1;
	bool shape_changed = false;
	if (multiplier < 1) {
		at_idx = tv[index];
		if (0 == multiplier) {
			multiplier = 1/(double) at_idx;
		}
		// edge indices
		if (0 == index) {
			// pop front (tv is not empty, otherwise in.n_dims() > 0 would catch it)
			tv.front() = std::move(tv.back());
			tv.pop_back();
			shape_changed = true;
		} else if (tv.size()-1 == index) {
			below_dim = ins.n_elems() / at_idx; // above is 1
			tv.pop_back();
			shape_changed = true;
		}
	} else if (multiplier >= 1 && index >= tv.size()) {
		at_idx = multiplier;
		// extending extra dimensions
		size_t extra_dims = index - tv.size();
		if (extra_dims) {
			tv.insert(tv.end(), extra_dims, 1);
		}
		tv.push_back(multiplier);
		below_dim = ins.n_elems() / at_idx;
		shape_changed = true;
	}

	if (false == shape_changed) {
		at_idx = tv[index];
		for (size_t i = 0; i < index; i++) {
			below_dim *= tv[i];
		}
		tv[index] *= multiplier;
		if (0 == tv[index]) {
			tv[index] = 1; // ensure the shape is defined if it was previously defined
		}
	}

	return tensor_shape(tv);
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
	tensor_shape news = transpose_shape(in.get_shape());
	memory_alloc all;
	std::vector<size_t> inl = news.as_list();
	size_t dimY = inl[0];
	size_t dimX = inl[1];
	tensor<T>* ans = new tensor<T>(all, news); // new shape
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
tensor<T>* ioperation<T>::extend_op (const tensor<T>& in, size_t index, size_t multiplier) const {
	size_t cline;
	size_t below_dim;
	tensor_shape ts = change_shape(in.get_shape(), index, multiplier, below_dim, cline); // new shape
	below_dim *= cline;

	memory_alloc all;
	tensor<T>* ans = new tensor<T>(all, ts);

	size_t above_dim = in.n_elems() / below_dim;
	// copy over data
	T* src_data = in.raw_data;
	T* dest_data = ans->raw_data;
	for (size_t i = 0; i < above_dim; i++) {
		// copy data multiplier times
		T* src_addr = src_data + i * below_dim;
		for (size_t j = 0; j < multiplier; j++) {
			T* dest_addr = dest_data + below_dim * (multiplier * i + j);
			std::memcpy(dest_addr, src_addr, below_dim * sizeof(T));
		}
	}

	return ans;
}

template <typename T>
tensor<T>* ioperation<T>::compress_op (
	const tensor<T>& in,
	size_t index,
	std::function<T(const std::vector<T>&)> collector) const {
	size_t total = in.n_elems();
	if (index >= in.n_dims()) {
		// compressing an non-existent dimension...
		tensor<T>* ans = new tensor<T>(in);
		return ans;
	}
	size_t cline;
	size_t below_dim;
	tensor_shape ts = change_shape(in.get_shape(), index, 0, below_dim, cline); // new shape

	memory_alloc all;
	tensor<T>* ans = new tensor<T>(all, ts);

	size_t above_dim = total / (below_dim*cline);
	// copy over data
	T* src_data = in.raw_data;
	T* dest_data = ans->raw_data;
	for (size_t i = 0; i < above_dim; i++) {
		for (size_t j = 0; j < below_dim; j++) {
			// apply compression to each element along cline dimension
			size_t dest_idx = j + i * below_dim;
			std::vector<T> gather;
			for (size_t k = 0; k < cline; k++) {
				size_t src_idx = j + k * below_dim + i * below_dim * cline;
				gather.push_back(src_data[src_idx]);
			}
			dest_data[dest_idx] = collector(gather);
		}
	}

	return ans;
}

// TODO consider options of not evaluating shape at construction
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
scalar<T>::scalar (const scalar<T>& other, std::string name) {
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

// UNARY MATRIX OPERATIONS

template <typename T>
void iunar_mat_ops<T>::copy (const ivariable<T>& other, std::string name) {
	if (const iunar_mat_ops<T>* uptr = dynamic_cast<const iunar_mat_ops<T>*>(&other)) {
		var = uptr->var;
	}
	ivariable<T>::copy(other, name);
}

template <typename T>
ivariable<T>& iunar_mat_ops<T>::operator () (ivariable<T>& var) {
	std::stringstream ns;
	ns << "<" << get_symb() << ">(" << var.get_name() << ")";
	this->name = ns.str();
	this->consume(var);
	this->var = &var;
	shape_eval();
	return *this;
}

template <typename T>
iunar_mat_ops<T>& iunar_mat_ops<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

// TENSOR EXTENSION

template <typename T>
tensor<T>* extend<T>::calc_derive (ivariable<T>* over) const {
	tensor<T>* ans = nullptr;
	if (this->var) {
		tensor<T>* deriv = this->var->derive(over);
		if (deriv) {
			// extend along index
			ans = this->extend_op(*deriv, index, multiplier);
			delete deriv;
		}
	}
	return ans;
}

template <typename T>
void extend<T>::shape_eval (void) {
	if (this->var) {
		tensor_shape ts = this->var->get_shape();
		if (ts.is_fully_defined()) {
			size_t below_dim;
			size_t cline;
			this->update(
				this->change_shape(ts, index, multiplier, below_dim, cline));
		}
	}
}

template <typename T>
void extend<T>::copy (const ivariable<T>& other, std::string name) {
	if (const extend<T>* eptr = dynamic_cast<const extend<T>*>(&other)) {
		index = eptr->index;
		multiplier = eptr->multiplier;
	}
	iunar_mat_ops<T>::copy(other, name);
}

template <typename T>
extend<T>::extend (ivariable<T>& in) : extend(in, 0, 1){}

template <typename T>
extend<T>::extend (ivariable<T>& in, size_t index, size_t multiplier)
	: index(index), multiplier(multiplier) {
	shape_eval();
	(*this)(in);
}

template <typename T>
void extend<T>::set_ext_info (size_t index, size_t multiplier) {
	this->index = index;
	this->multiplier = multiplier;
	shape_eval(); // re-eval
}

template <typename T>
extend<T>* extend<T>::clone (std::string name) {
	return new extend<T>(*this, name);
}

template <typename T>
extend<T>& extend<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
const tensor<T>& extend<T>::eval (void) {
	assert(nullptr != this->var);
	const tensor<T>& in = this->var->eval();
	tensor<T>* ans = this->extend_op(in, index, multiplier);
	this->out = *ans;
	delete ans;
	return this->out;
}

// TENSOR COMPRESSION

template <typename T>
tensor<T>* compress<T>::calc_derive (ivariable<T>* over) const {
	tensor<T>* ans = nullptr;
	if (this->var) {
		tensor<T>* deriv = this->var->derive(over);
		if (deriv) {
			// uncompress along index
			ans = this->compress_op(*deriv, index, collector);
			delete deriv;
		}
	}
	return ans;
}

template <typename T>
void compress<T>::shape_eval (void) {
	if (this->var) {
		tensor_shape ts = this->var->get_shape();
		if (ts.is_fully_defined()) {
			size_t below_dim;
			size_t cline;
			this->update(
				this->change_shape(ts, index, 0, below_dim, cline));
		}
	}
}

template <typename T>
void compress<T>::copy (const ivariable<T>& other, std::string name) {
	if (const compress<T>* cptr = dynamic_cast<const compress<T>*>(&other)) {
		index = cptr->index;
		collector = cptr->collector;
	}
	iunar_mat_ops<T>::copy(other, name);
}

template <typename T>
compress<T>::compress (ivariable<T>& in) : compress(in, 0) {}

template <typename T>
compress<T>::compress (ivariable<T>& in, size_t index) : compress(in, index, [](const std::vector<T>& data) {
	T ans = 0;
	for (T raw : data) {
		ans += raw;
	}
	ans /= data.size();
	return ans;
}) {}

template <typename T>
compress<T>::compress (ivariable<T>& in, size_t index, std::function<T(const std::vector<T>&)> collector)
	: index(index), collector(collector) {
	shape_eval();
	(*this)(in);
}

template <typename T>
void compress<T>::set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector) {
	this->index = index;
	this->collector = collector;
	shape_eval(); // re-eval
}

template <typename T>
compress<T>* compress<T>::clone (std::string name) {
	return new compress<T>(*this, name);
}

template <typename T>
compress<T>& compress<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
const tensor<T>& compress<T>::eval (void) {
	assert(nullptr != this->var);
	const tensor<T>& in = this->var->eval();
	tensor<T>* ans = this->compress_op(in, index, collector);
	this->out = *ans;
	delete ans;
	return this->out;
}

// MATRIX TRANSPOSE

template <typename T>
tensor<T>* transpose<T>::calc_derive (ivariable<T>* over) const {
	tensor<T>* ans = nullptr;
	if (this->var) {
		tensor<T>* deriv = this->var->derive(over);
		if (deriv) {
			ans = this->transpose_op(*deriv); // transpose
			delete deriv;
		}
	}
	return ans;
}

template <typename T>
void transpose<T>::shape_eval (void) {
	if (this->var) {
		tensor_shape ts = this->var->get_shape();
		if (ts.is_fully_defined()) {
			this->update(this->transpose_shape(ts));
		}
	}
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
const tensor<T>& transpose<T>::eval (void) {
	assert(nullptr != this->var);
	const tensor<T>& in = this->var->eval();
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
	if (a && b &&
		a->get_shape().is_fully_defined() &&
		b->get_shape().is_fully_defined()) {
		size_t common_dim;
		tensor_shape ts = this->get_matrix_shape(
			this->get_eval(*this->a),
			this->get_eval(*this->b),
			transposeA, transposeB, common_dim);
		assert(ts.is_fully_defined()); // assert initial shape is at least valid (re-checked at eval time)
		this->update(ts);
	}
}

template <typename T>
matmul<T>::matmul (const matmul<T>& other, std::string name) {
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
matmul<T>& matmul<T>::operator = (const ivariable<T>& other) {
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

}

#endif
