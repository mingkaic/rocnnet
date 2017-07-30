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
		this->notify(UPDATE);
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
		this->notify(UPDATE);
	}
	return *this;
}

template <typename T>
varptr<T> ivariable<T>::derive (inode<T>* wrt)
{
	if (this == wrt)
	{
		return constant<T>::get_shared_one();
	}
	return constant<T>::get_shared_zero();
}

template <typename T>
bool ivariable<T>::can_init (void) const
{
	return init_ != nullptr;
}

template <typename T>
ivariable<T>::ivariable (const tensorshape& shape,
	initializer<T>* init,
	std::string name) :
ileaf<T>(shape, name), init_(init) {}

template <typename T>
ivariable<T>::ivariable (const ivariable<T>& other) :
	ileaf<T>(other)
{
	copy_helper(other);
}

template <typename T>
ivariable<T>::ivariable (ivariable<T>&& other) :
	ileaf<T>(std::move(other))
{
	move_helper(std::move(other));
}

template <typename T>
void ivariable<T>::copy_helper (const ivariable<T>& other)
{
	if (init_ == other.init_) return;
	if (nullptr != init_)
	{
		delete init_;
	}
	if (other.init_)
	{
		init_ = other.init_->clone();
	}
	else
	{
		init_ = nullptr;
	}
}

template <typename T>
void ivariable<T>::move_helper (ivariable<T>&& other)
{
	if (init_ == other.init_) return;
	if (nullptr != init_)
	{
		delete init_;
	}
	init_ = std::move(other.init_);
	other.init_ = nullptr;
}

}

#endif
