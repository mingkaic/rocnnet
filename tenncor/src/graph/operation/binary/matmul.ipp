//
//  matmul.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef matop_hpp

namespace nnet {

// MATRIX MULTIPLICATION

template <typename T>
void matmul<T>::setup_gradient (void) {
	// matmul'(f, g) = inherited(matmul(k, g^T) * f' + matmul(f^T, k) * g')
	ivariable<T>* arga = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	ivariable<T>* argb = dynamic_cast<ivariable<T>*>(this->dependencies_[1]);
	assert(arga && argb);
	this->grad_ = new jacobian<T>([arga, argb, this](ivariable<T>* channel) {
		ivariable<T>* ga = arga->get_gradient();
		ivariable<T>* gb = argb->get_gradient();
		return new matmul<T>(channel, argb, transposeA, !transposeB) * ga +
				new matmul<T>(arga, channel, !transposeA, transposeB) * gb;
	}
}

template <typename T>
void matmul<T>::shape_eval (void) {
	if (a && b &&
		a->get_shape().is_fully_defined() &&
		b->get_shape().is_fully_defined()) {
		size_t common_dim;
		tensor_shape ts = this->get_matrix_shape(
				this->get_tensor_from(a),
				this->get_tensor_from(b),
				transposeA, transposeB, common_dim);
		assert(ts.is_fully_defined()); // assert initial shape is at least valid (re-checked at eval time)
		this->update(ts);
	}
}

template <typename T>
matmul<T>::matmul (const matmul<T>& other, std::string name) {
	transposeA = other.transposeA;
	transposeB = other.transposeB;
	ioperation<T>::copy(other, name);
}

template <typename T>
matmul<T>::matmul (ivariable<T>* a, ivariable<T>* b,
					bool transposeA, bool transposeB) :
		ioperation<T>(std::vector<ivariable<T>*>{a, b},
		nnutils::formatter() << "(" << a->get_name() << "•" 
		<< b->get_name() << ")"), transposeA_(transposeA), transposeB_(transposeB) {
}

template <typename T>
ievoker<T>* matmul<T>::clone_impl (std::string name) {
	return new matmul(*this, name);
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
void matmul<T>::update (void) {
	ivariable<T>* a = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	ivariable<T>* b = dynamic_cast<ivariable<T>*>(this->dependencies_[1]);
	const tensor<T>& at = a->get_eval();
	const tensor<T>& bt = b->get_eval();
	tensor<T>* ans = this->matmul_op(at, bt, transposeA, transposeB);
	this->out_ = *ans;
	delete ans;
	this->notify();
}

}

#endif