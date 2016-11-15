//
//  tensor_jacobi.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef tensor_jacobi_hpp

#include "graph/operation/special/matmul.hpp"

namespace nnet
{

template <typename T>
void tensor_jacobi<T>::clear_ownership (void)
{
	owner_.clear();
	root_ = nullptr;
}

template <typename T>
void tensor_jacobi<T>::copy (const tensor_jacobi<T>& other)
{
	transposeA_ = other.transposeA_;
	transposeB_ = other.transposeB_;
	k_ = other.k_;
	owner_ = other.owner_;
	root_ = other.root_;
	tensor<T>::copy(other);
}

template <typename T>
tensor_jacobi<T>::tensor_jacobi (const tensor_jacobi<T>& other)
{
	copy(other);
}

template <typename T>
tensor<T>* tensor_jacobi<T>::clone_impl (void)
{
    return new tensor_jacobi<T>(*this);
}

template <typename T>
T* tensor_jacobi<T>::get_raw (void)
{
	tensor<T>* src = root_->get_eval();
	copy(*src);
	return tensor<T>::get_raw();
}

template <typename T>
tensor_jacobi<T>::tensor_jacobi (bool transposeA, bool transposeB) :
	transposeA_(transposeA), transposeB_(transposeB) {}

template <typename T>
tensor_jacobi<T>::~tensor_jacobi (void) { clear_ownership(); }

// COPY
template <typename T>
tensor_jacobi<T>* tensor_jacobi<T>::clone (void)
{
    return static_cast<tensor_jacobi<T>*>(clone_impl());
}

template <typename T>
tensor_jacobi<T>& tensor_jacobi<T>::operator = (const tensor_jacobi<T>& other)
{
    if (this != &other)
    {
    	copy(other);
    }
    return *this;
}

template <typename T>
void tensor_jacobi<T>::set_root (ivariable<T>* root) { k_ = root; }

template <typename T>
const tensor_jacobi<T>& tensor_jacobi<T>::operator () (
    ivariable<T>* arga, 
    ivariable<T>* argb)
{
	// J(a, b) = d(a) * matmul(k, b^T) + d(b) * matmul(a^T, k)
	clear_ownership();
	// we're at evaluation time, so we can assess which derivative to take
	const tensor<T>* ag = arga->get_gradient()->get_eval();
	const tensor<T>* bg = argb->get_gradient()->get_eval();
	ivariable<T>* a = nullptr;
	if (ag)
	{
		a = new matmul<T>(k_, argb, transposeA_, !transposeB_);
		owner_.push_back(std::shared_ptr<ivariable<T> >(a));
		a = a * arga->get_gradient();
	}
	ivariable<T>* b = nullptr;
	if (bg)
	{
		b = new matmul<T>(arga, k_, !transposeA_, transposeB_);
		owner_.push_back(std::shared_ptr<ivariable<T> >(b));
		b = b * argb->get_gradient();
	}
	root_ = a + b;
	return *this;
}

}

#endif