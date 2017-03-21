//
//  ivariable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-27.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_IVARIABLE_HPP

namespace nnet
{

template <typename T>
ivariable<T>::~ivariable (void)
{
	if (nullptr != init_)
	{
		delete init_;
	}
}

template <typename T>
ivariable<T>* ivariable<T>::clone (void) const
{
	return static_cast<ivariable<T>*>(this->clone_impl());
}

template <typename T>
ivariable<T>* ivariable<T>::move (void)
{
	return static_cast<ivariable<T>*>(this->move_impl());
}

template <typename T>
ivariable<T>& ivariable<T>::operator = (const ivariable<T>& other)
{
	if (this != &other)
	{
		ileaf<T>::operator = (other);
		copy_helper(other);
	}
	return *this;
}

template <typename T>
ivariable<T>& ivariable<T>::operator = (ivariable<T>&& other)
{
	if (this != &other)
	{
		ileaf<T>::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

template <typename T>
bool ivariable<T>::can_init (void) const
{
	return init_ != nullptr;
}

template <typename T>
const tensor<T>* ivariable<T>::get_gradient (inode<T>* wrt)
{
	if (this == wrt)
	{
		return one->get_eval();
	}
	return zero->get_eval();
}

template <typename T>
ivariable<T>::ivariable (const tensorshape& shape,
	initializer<T>* init,
	std::string name) :
ileaf<T>(shape, name), init_(init)
{
	common();
}

template <typename T>
ivariable<T>::ivariable (const ivariable<T>& other) :
	ileaf<T>(other)
{
	common();
	copy_helper(other);
}

template <typename T>
ivariable<T>::ivariable (ileaf<T>&& other) :
	ileaf<T>(other)
{
	common();
	move_helper(std::move(other));
}

template <typename T>
void ivariable<T>::copy_helper (const ivariable<T>& other)
{
	if (nullptr != init_)
	{
		delete init_;
	}
	init_ = other.init_->clone();
}

template <typename T>
void ivariable<T>::move_helper (ivariable<T>&& other)
{
	if (nullptr != init_)
	{
		delete init_;
	}
	init_ = std::move(other.init_);
}

template <typename T>
void ivariable<T>::common (void)
{
	one = std::unique_ptr<constant<T> >(constant<T>::get(0));
	zero = std::unique_ptr<constant<T> >(constant<T>::get(0));
	one->is_managed_ = true;
	zero->is_managed_ = true;
}

}

#endif
