//
//  buffer.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef buffer_hpp

namespace nnet
{

// BUFFER
template <typename T>
buffer<T>::buffer (const buffer<T>& other, std::string name) :
	iconnector<T>(other, name) {}

template <typename T>
ivariable<T>* buffer<T>::clone_impl (std::string name)
{
	return new buffer(*this, name);
}

template <typename T>
buffer<T>::buffer (ivariable<T>* leaf, std::string name) :
	iconnector<T>(std::vector<ivariable<T>*>{leaf}, name) {}

template <typename T>
buffer<T>* buffer<T>::clone (std::string name)
{
	return static_cast<buffer<T>*>(clone_impl(name));
}

template <typename T>
buffer<T>& buffer<T>::operator = (const buffer<T>& other)
{
	if (this != &other)
	{
		iconnector<T>::copy(other);
	}
	return *this;
}

template <typename T>
buffer<T>& buffer<T>::operator = (ivariable<T>& ivar)
{
	// change dependency
	change_dep(&ivar);
	return *this;
}

template <typename T>
ivariable<T>* buffer<T>::get (void) const
{
	return sub_to_var<T>(this->dependencies_[0]);
}

template <typename T>
tensorshape buffer<T>::get_shape (void) const
{
	return get()->get_shape();
}

template <typename T>
tensor<T>* buffer<T>::get_eval (void)
{
	return get()->get_eval();
}

template <typename T>
ivariable<T>* buffer<T>::get_gradient (void)
{
	return get()->get_gradient();
}

}

#endif