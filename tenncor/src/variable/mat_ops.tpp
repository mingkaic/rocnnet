//
// Created by Mingkai Chen on 2016-10-08.
//

#ifdef matop_hpp

namespace nnet {

// TENSOR EXTENSION

template <typename T>
tensor<T>* extend<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
	tensor<T>* ans = nullptr;
	if (this->var) {
		tensor<T>* deriv = this->var->gradient(over);
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
	iunar_ops<T>::copy(other, name);
}

template <typename T>
extend<T>::extend (VAR_PTR<T> in) : extend(in, 0, 1){}

template <typename T>
extend<T>::extend (VAR_PTR<T> in, WEAK_VAR_PTR<T> watch) : watch(watch) {
	this->consume(*(watch.lock().get()));
	shape_eval();
	(*this)(in);
}

template <typename T>
extend<T>::extend (VAR_PTR<T> in, size_t index, size_t multiplier)
		: index(index), multiplier(multiplier) {
	shape_eval();
	(*this)(in);
}

template <typename T>
EVOKER_PTR<T> extend<T>::clone_impl (std::string name) {
	return std::shared_ptr<extend<T> >(new extend(*this, name));
}

template <typename T>
extend<T>& extend<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
void extend<T>::set_ext_info (WEAK_VAR_PTR<T> watch) {
	this->index = 0;
	this->multiplier = 0;
	this->consume(*(watch.get()));
	this->watch = watch;
}

template <typename T>
void extend<T>::set_ext_info (size_t index, size_t multiplier) {
	this->index = index;
	this->multiplier = multiplier;
	watch = nullptr;
	std::unordered_set<ioperation<T>*> cons = watch->get_consumers();
	if (cons.end() != cons.find(this)) {
		cons.erase(this);
	}
}

template <typename T>
const tensor<T>& extend<T>::eval (void) {
	assert(nullptr != this->var);
	const tensor<T>& in = this->var->eval();
	if (nullptr == watch.lock()) {
		tensor<T> *ans = this->extend_op(in, index, multiplier);
		this->out = *ans;
		delete ans;
	} else {
		this->out = in;
		std::vector<size_t> target = watch.lock()->get_shape().as_list();
		std::vector<size_t> orig = in.get_shape().as_list();
		// this actually get expensive for big shapes, TODO: refactor/change anti-pattern
		for (size_t i = 0; i < target.size(); i++) {
			if (i < orig.size()) {
				if (target[i] > orig[i]) {
					tensor<T> *ans = this->extend_op(in, i, orig[i]/target[i]);
					this->out = *ans;
					delete ans;
				}
			} else {
				tensor<T>* ans = this->extend_op(in, i, target[i]);
				this->out = *ans;
				delete ans;
			}
		}
	}
	return this->out;
}

// TENSOR COMPRESSION

template <typename T>
tensor<T>* compress<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
	tensor<T>* ans = nullptr;
	if (this->var) {
		tensor<T>* deriv = this->var->gradient(over);
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
	iunar_ops<T>::copy(other, name);
}

template <typename T>
compress<T>::compress (VAR_PTR<T> in) : compress(in, 0) {}

template <typename T>
compress<T>::compress (VAR_PTR<T> in, size_t index) : compress(in, index, [](const std::vector<T>& data) {
	T ans = 0;
	for (T raw : data) {
		ans += raw;
	}
	ans /= data.size();
	return ans;
}) {}

template <typename T>
compress<T>::compress (VAR_PTR<T> in, size_t index, std::function<T(const std::vector<T>&)> collector)
		: index(index), collector(collector) {
	shape_eval();
	(*this)(in);
}

template <typename T>
EVOKER_PTR<T> compress<T>::clone_impl (std::string name) {
	return std::shared_ptr<compress<T> >(new compress(*this, name));
}

template <typename T>
compress<T>& compress<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
void compress<T>::set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector) {
	this->index = index;
	this->collector = collector;
	shape_eval(); // re-eval
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
tensor<T>* transpose<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
	tensor<T>* ans = nullptr;
	if (this->var) {
		tensor<T>* deriv = this->var->gradient(over);
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
transpose<T>::transpose (VAR_PTR<T> in) {
	(*this)(in);
}

template <typename T>
EVOKER_PTR<T> transpose<T>::clone_impl (std::string name) {
	return std::shared_ptr<transpose<T> >(new transpose(*this, name));
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
tensor<T>* matmul<T>::calc_gradient (WEAK_VAR_PTR<T> over) const {
	tensor<T>* ans = nullptr;
	if (a == over.lock()) {
		// we take the trace of b
		ans = this->get_trace(*b);
	} else if (b == over.lock()) {
		// we take the trace of a
		ans = this->get_trace(*a);
	} else {
		tensor<T>* deriva = a->gradient(over);
		tensor<T>* derivb = b->gradient(over);
		ans = this->util_op(*deriva, *derivb, shared_cnnet::op_add<T>);
		delete deriva;
		delete derivb;
	}
	return ans;
}

template <typename T>
void matmul<T>::shape_eval (void) {
	if (a && b &&
		a->get_shape().is_fully_defined() &&
		b->get_shape().is_fully_defined()) {
		size_t common_dim;
		tensor_shape ts = this->get_matrix_shape(
				this->get_eval(this->a),
				this->get_eval(this->b),
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
matmul<T>::matmul (VAR_PTR<T> a, VAR_PTR<T> b,
					bool transposeA, bool transposeB) {
	(*this)(a, b, transposeA, transposeB);
}

template <typename T>
EVOKER_PTR<T> matmul<T>::clone_impl (std::string name) {
	return std::shared_ptr<matmul<T> >(new matmul(*this, name));
}

template <typename T>
ivariable<T>& matmul<T>::operator () (VAR_PTR<T> a, VAR_PTR<T> b,
										bool transposeA, bool transposeB) {
	std::stringstream ns;
	ns << a->get_name() << "•" << b->get_name();
	this->name = ns.str();
	this->consume(*(a.get())); this->consume(*(b.get()));
	this->a = a;
	this->b = b;
	this->transposeA = transposeA;
	this->transposeB = transposeB;
	size_t common_dim;
	if (session::pre_shape_eval()) {
		shape_eval();
	}
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