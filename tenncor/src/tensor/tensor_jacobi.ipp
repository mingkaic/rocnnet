//
//  tensor_jacobi.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef tensor_jacobi_hpp

namespace nnet
{

template <typename T, typename A>
void tensor_jacobi<T,A>::clear_ownership (void)
{
	owner_.clear();
	root_ = nullptr;
}

template <typename T, typename A>
void tensor_jacobi<T,A>::copy (const tensor_jacobi<T,A>& other)
{
	transposeA_ = other.transposeA_;
	transposeB_ = other.transposeB_;
	k_ = other.k_;
	owner_ = other.owner_;
	root_ = other.root_;
	tensor<T,A>::copy(other);
}

template <typename T, typename A>
tensor_jacobi<T,A>::tensor_jacobi (const tensor_jacobi<T,A>& other)
{
	copy(other);
}

template <typename T, typename A>
tensor<T,A>* tensor_jacobi<T,A>::clone_impl (void)
{
    return new tensor_jacobi<T,A>(*this);
}

template <typename T, typename A>
T* tensor_jacobi<T,A>::get_raw (void)
{
	tensor<T,A>* src = root_->get_eval();
	tensor<T,A>::copy(*src);
	return tensor<T,A>::get_raw();
}

template <typename T, typename A>
tensor_jacobi<T,A>::tensor_jacobi (bool transposeA, bool transposeB) :
	transposeA_(transposeA), transposeB_(transposeB) {}

template <typename T, typename A>
tensor_jacobi<T,A>::~tensor_jacobi (void) { clear_ownership(); }

// COPY

template <typename T, typename A>
tensor_jacobi<T,A>* tensor_jacobi<T,A>::clone (void)
{
    return static_cast<tensor_jacobi<T,A>*>(clone_impl());
}

template <typename T, typename A>
tensor_jacobi<T,A>& tensor_jacobi<T,A>::operator = (const tensor_jacobi<T,A>& other)
{
    if (this != &other)
    {
    	copy(other);
    }
    return *this;
}

template <typename T, typename A>
void tensor_jacobi<T,A>::set_root (ivariable<T>* root) { k_ = root; }

template <typename T, typename A>
const tensor_jacobi<T,A>& tensor_jacobi<T,A>::operator () (
    ivariable<T>* arga, 
    ivariable<T>* argb)
{
	// J(a, b) = d(a) * matmul(k, b^T) + d(b) * matmul(a^T, k)
	clear_ownership();
	// we're at evaluation time, so we can assess which derivative to take
	const tensor<T,A>* ag = arga->get_gradient()->get_eval();
	const tensor<T,A>* bg = argb->get_gradient()->get_eval();
	varptr<T> a;
	if (ag)
	{
		a = matmul<T>::build(k_, argb, transposeA_, !transposeB_);
		owner_.push_back(std::shared_ptr<ivariable<T> >(a.get()));
		a = a * varptr<double>(arga->get_gradient());
	}
	varptr<T> b;
	if (bg)
	{
		b = matmul<T>::build(arga, k_, !transposeA_, transposeB_);
		owner_.push_back(std::shared_ptr<ivariable<T> >(b.get()));
		b = b * varptr<double>(argb->get_gradient());
	}
	root_ = a + b;
	return *this;
}

}

#endif