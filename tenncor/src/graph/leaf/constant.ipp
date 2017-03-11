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
void* constant<T>::operator new (size_t size, T scalar)
{
	constant<T>* c = static_cast<constant<T>*>(
		::operator new(size, scalar));
	c->onheap_ = true;
	return c;
}

template <typename T>
void* constant<T>::operator new (size_t size, std::vector<T> raw, tensorshape shape)
{
	constant<T>* c = static_cast<constant<T>*>(
		::operator new(size, raw, shape));
	c->onheap_ = true;
	return c;
}

template <typename T>
constant<T>::constant (T scalar) :
	ileaf<T>(std::vector<size_t>{1},
		 new const_init<T>(scalar),
		 nnutils::formatter() << scalar),
	zero(0)
{
	this->data_->allocate();
	(*this->init_)(*this->data_);
	this->is_init_ = true;
}

template <typename T>
constant<T>::constant (std::vector<T> raw, tensorshape shape) :
	ileaf<T>(shape, new typename ileaf<T>::assignment(),
		 nnutils::formatter() << raw.front() << ".."
		 	<< raw.back() << raw.end()),
	zero(0)
{
	this->data_->allocate();
	(*this->init_) = raw;
	this->is_init_ = true;
}

template <typename T>
constant<T>* constant<T>::clone (void) const
{
	return static_cast<constant<T>*>(clone_impl());
}

template <typename T>
constant<T>::constant (constant<T>&& other) :
	ileaf<T>(other), zero(0)  {}

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
		ileaf<T>::operator = (other);
	}
	return *this;
}

template <typename T>
const tensor<T>* constant<T>::get_gradient (inode<T>* wrt) const
{
	return this->zero->get_eval();
}

template <typename T>
void constant<T>::commit_sudoku (void)
{
	if (onheap_)
	{
		delete this;
	}
}

template <typename T>
constant<T>::constant (const constant<T>& other) :
	ileaf<T>(other), zero(0) {}

template <typename T>
inode<T>* constant<T>::clone_impl (void) const
{
	return new constant<T>(*this);
}

template <typename T>
void constant<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const {}

template <typename T>
inode<T>* constant<T>::get_leaf (variable<T>* leaf) const
{
	return &this->zero;
}

}

#endif