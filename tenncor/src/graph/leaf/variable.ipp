//
//  leaf.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_VARIABLE_HPP

namespace nnet
{

template <typename T>
variable<T>::variable (T scalar, std::string name) :
	ivariable<T>(std::vector<size_t>{1},
		new const_init<T>(scalar), name)
{
	initialize();
}

template <typename T>
variable<T>::variable (const tensorshape& shape, std::string name) :
	ivariable<T>(shape, nullptr, name) {}

template <typename T>
variable<T>::variable (const tensorshape& shape,
	const initializer<T>& init, std::string name) :
ivariable<T>(shape, init.clone(), name) {}

template <typename T>
variable<T>* variable<T>::clone (void) const
{
	return static_cast<variable<T>*>(clone_impl());
}

template <typename T>
variable<T>::variable (variable<T>&& other) :
	ivariable<T>(other) {}

template <typename T>
variable<T>& variable<T>::operator = (const variable<T>& other)
{
	if (this != &other)
	{
		ileaf<T>::operator = (other);
	}
	return *this;
}

template <typename T>
variable<T>& variable<T>::operator = (variable<T>&& other)
{
	if (this != &other)
	{
		ileaf<T>::operator = (other);
	}
	return *this;
}

template <typename T>
void variable<T>::set_initializer (const initializer<T>& init)
{
	if (this->init_)
	{
		delete this->init_;
	}
	this->init_ = init.clone();
}

template <typename T>
tensor<T>& variable<T>::initialize (void)
{
	assert(nullptr != this->init_);
	// if not alloc, attempt to allocate
	if (false == this->data_->is_alloc() &&
		this->data_->allocate())
	{
		throw std::exception(); // todo: better exception
	}
	(*this->init_)(*this->data_);
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensorshape shape)
{
	assert(this->init_ != nullptr);
	if (false == this->data_->is_alloc() &&
		this->data_->allocate(shape))
	{
		throw std::exception(); // todo: better exception
	}
	(*this->init_)(*this->data_);
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

template <typename T>
inode<T>* variable<T>::get_leaf (variable<T>* leaf)
{
	if (this == leaf)
	{
		return this->one.get();
	}
	return this->zero.get();
}

template <typename T>
void variable<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const
{
	leaves.emplace(const_cast<variable<T>*>(this), nullptr);
}

template <typename T>
variable<T>::variable (const variable<T>& other) :
	ivariable<T>(other) {}

template <typename T>
inode<T>* variable<T>::clone_impl (void) const
{
	return new variable(*this);
}

}

#endif
