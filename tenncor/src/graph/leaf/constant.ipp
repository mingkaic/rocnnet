//
//  constant.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_CONSTANT_HPP

namespace nnet
{

template <typename T>
constant<T>* constant<T>::get (T scalar)
{
	return new constant<T>(scalar);
}

template <typename T>
constant<T>* constant<T>::get (std::vector<T> raw, tensorshape shape)
{
	return new constant<T>(raw, shape);
}

template <typename T>
constant<T>* constant<T>::get (constant<T>&& other)
{
	return new constant<T>(std::move(other));
}

template <typename T>
constant<T>::~constant (void)
{
	if (zero != this)
	{
		delete zero;
	}
}

template <typename T>
constant<T>* constant<T>::clone (void) const
{
	return static_cast<constant<T>*>(clone_impl());
}

template <typename T>
constant<T>& constant<T>::operator = (const constant<T>& other)
{
	if (this != &other)
	{
		ileaf<T>::operator = (other);
	}
	return *this;
}

template <typename T>
constant<T>& constant<T>::operator = (constant<T>&& other)
{
	if (this != &other)
	{
		ileaf<T>::operator = (std::move(other));
	}
	return *this;
}

template <typename T>
const tensor<T>* constant<T>::get_gradient (inode<T>* wrt) const
{
	return zero->get_eval();
}

template <typename T>
inode<T>* constant<T>::get_leaf (variable<T>* leaf)
{
	return zero;
}

template <typename T>
void constant<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const {}

template <typename T>
constant<T>::constant (T scalar) :
	ileaf<T>(std::vector<size_t>{1},
		 nnutils::formatter() << scalar)
{
	if (scalar == 0)
	{
		zero = this;
	}
	else
	{
		zero = new constant<T>(0);
		zero->is_managed_ = true;
	}

	const_init<T> init(scalar);
	this->data_->allocate();
	init(*this->data_);
	this->is_init_ = true;
}

template <typename T>
constant<T>::constant (std::vector<T> raw, tensorshape shape) :
	ileaf<T>(shape, nnutils::formatter()
		<< raw.front() << ".." << raw.back())
{
	zero = new constant<T>(0);
	zero->is_managed_ = true;

	typename ileaf<T>::assignment assigner;
	this->data_->allocate();
	assigner(*this->data_, raw);
	this->is_init_ = true;
}

template <typename T>
void constant<T>::commit_sudoku (void)
{
	if (false == is_managed_ && this->no_audience())
	{
		delete this;
	}
}

template <typename T>
constant<T>::constant (const constant<T>& other) :
	ileaf<T>(other)
{
	zero = new constant<T>(0);
	zero->is_managed_ = true;
}

template <typename T>
constant<T>::constant (constant<T>&& other) :
	ileaf<T>(std::move(other))
{
	zero = new constant<T>(0);
	zero->is_managed_ = true;
}

template <typename T>
inode<T>* constant<T>::clone_impl (void) const
{
	return new constant<T>(*this);
}

}

#endif