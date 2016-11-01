//
//  matmul.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef matop_hpp
#include <iostream>

namespace nnet {

// MATRIX MULTIPLICATION

template <typename T>
void matmul<T>::make_gradient (VAR_PTR<T>& safety_ref) {
	// same as multiplication
	// matmul'(f, g) = matmul(f',g) + matmul(f,g')
	VAR_PTR<T> ga = this->a->get_gradient();
	VAR_PTR<T> gb = this->b->get_gradient();
	// force ga and ba to conform to their respective shapes
	VAR_PTR<T> ea = extend<T>::make(ga, a);
	VAR_PTR<T> eb = extend<T>::make(gb, b);

	VAR_PTR<T> ma = matmul<T>::make(ea, b, transposeA, transposeB);
	VAR_PTR<T> mb = matmul<T>::make(a, eb, transposeA, transposeB);
	this->set_gradient(ma + mb);
	safety_ref = this->grad;
}

template <typename T>
void matmul<T>::shape_eval (void) {
	if (a && b &&
		a->get_shape().is_fully_defined() &&
		b->get_shape().is_fully_defined()) {
		size_t common_dim;
		tensor_shape ts = this->get_matrix_shape(
				this->get_eval(a),
				this->get_eval(b),
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
}

template <typename T>
EVOKER_PTR<T> matmul<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new matmul(*this, name));
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
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	assert(nullptr != a && nullptr != b);
	const tensor<T>& at = this->a->eval();
	const tensor<T>& bt = this->b->eval();
	tensor<T>* ans = this->matmul_op(at, bt, transposeA, transposeB);
	this-> out_ = *ans;
	delete ans;
	return this-> out_;
}

}

#endif