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
ivariable<T>::ivariable (const ivariable<T>& other) :
	subject_owner(other),
	id_(r_temp::temp_uuid(this))
{
	name_ = other.name_;
}

template <typename T>
ivariable<T>::ivariable (std::string name) :
	ccoms::subject_owner(),
	id_(r_temp::temp_uuid(this)),
	name_(name)
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
ivariable<T>& ivariable<T>::operator = (const ivariable<T>& other)
{
	if (this != &other)
	{
		subject_owner::operator = (other);
		name_ = other.name_;
	}
	return *this;
}

}

#endif