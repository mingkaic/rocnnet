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
	if (0 == name.size())
	{
		name_ = other.name_+"_cpy";
	}
	else
	{
		name_ = name;
	}
	ccoms::subject_owner::copy(other);
}

template <typename T>
ivariable<T>::ivariable (const ivariable<T>& other, std::string name)
{
	copy(other, name);
}

template <typename T>
ivariable<T>::ivariable (std::string name) :
	ccoms::subject_owner(), name_(name)
{
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