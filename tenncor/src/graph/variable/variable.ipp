//
//  variable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef variable_hpp

namespace nnet
{

// VARIABLE IMPLEMENTATION

template <typename T>
variable<T>::variable (T scalar, std::string name) :
	ileaf<T>(std::vector<size_t>{1},
	new const_init<T>(scalar), name)
{
	initialize();
}

template <typename T>
variable<T>::variable (const tensorshape& shape, std::string name) :
	ileaf<T>(shape, nullptr, name) {}

template <typename T>
variable<T>::variable (const tensorshape& shape, initializer<T>& init, std::string name) :
	ileaf<T>(shape, init.clone(), name) {}

template <typename T>
variable<T>* variable<T>::clone (void)
{
	return new variable<T>(*this);
}

template <typename T>
void variable<T>::set_initializer (initializer<T>& init)
{
	this->init_ = init.clone();
}

template <typename T>
tensor<T>& variable<T>::initialize (void)
{
	assert(this->init_ != nullptr);
	if (false == this->out_->is_alloc())
	{ // if not alloc, allocate
		this->out_->allocate();
	}
	(*this->init_)(*this->out_);
	this->is_init_ = true;
	this->notify();
	return *this->out_;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensorshape alloc_shape)
{
	assert(this->init_ != nullptr);
	if (false == this->out_->is_alloc())
	{ // if not alloc, allocate
		this->out_->allocate(alloc_shape);
	}
	(*this->init_)(*this->out_);
	this->is_init_ = true;
	this->notify();
	return *this->out_;
}

}

#endif
