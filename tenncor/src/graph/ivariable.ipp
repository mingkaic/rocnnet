//
//  ivariable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef ivariable_hpp

namespace nnet
{

// VARIABLE INTERFACE

template <typename T>
void ivariable<T>::copy (const ivariable<T>& other, std::string name)
{
	out_ = std::unique_ptr<tensor<T> >(other.out_->clone());
	if (0 == name.size())
	{
		name_ = other.name_+"_cpy";
	}
	else
	{
		name_ = name;
	}
}

template <typename T>
ivariable<T>::ivariable (const ivariable<T>& other, std::string name) :
	ccoms::subject(other)
{
	copy(other, name);
}

template <typename T>
ivariable<T>::ivariable (const tensorshape& shape, std::string name) : 
	name_(name)
{
	if (shape.is_fully_defined())
	{
		out_ = std::make_unique<tensor<T> >(shape);
	}
	session& sess = session::get_instance();
	sess.register_obj(*this);
}

template <typename T>
ivariable<T>::~ivariable (void)
{
	session& sess = session::get_instance();
	sess.unregister_obj(*this);
}

template <typename T>
ivariable<T>* ivariable<T>::clone (std::string name)
{
	return clone_impl(name);
}

template <typename T>
ivariable<T>& ivariable<T>::operator = (const ivariable<T>& other)
{
	if (this != &other)
	{
		copy(other);
	}
	return *this;
}

}

#endif