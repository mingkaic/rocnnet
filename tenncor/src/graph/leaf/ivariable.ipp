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
ivariable<T>* ivariable<T>::clone (void) const
{
	return static_cast<ivariable<T>*>(this->clone_impl());
}

template <typename T>
const tensor<T>* ivariable<T>::get_gradient (inode<T>* wrt) const
{
	if (this == wrt)
	{
		return this->one->get_eval();
	}
	return this->zero->get_eval();
}

template <typename T>
ivariable<T>::ivariable (const tensorshape& shape,
	itensor_handler<T>* init,
	std::string name) :
ileaf<T>(shape, init, name), zero(0), one(1) {}

template <typename T>
ivariable<T>::ivariable (const ileaf<T>& other) :
	ileaf<T>(other) {}

template <typename T>
ivariable<T>::ivariable (ileaf<T>&& other) :
	ileaf<T>(other) {}

}

#endif
