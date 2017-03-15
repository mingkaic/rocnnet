//
//  placeholder.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_PLACEHOLDER_HPP

namespace nnet
{

template <typename T>
placeholder<T>::placeholder (const tensorshape& shape, std::string name) :
	ivariable<T>(shape, new typename ileaf<T>::assignment(), name) {}

template <typename T>
placeholder<T>* placeholder<T>::clone (void) const
{
	return static_cast<placeholder<T>*>(clone_impl());
}

template <typename T>
placeholder<T>::placeholder (placeholder<T>&& other) :
	ivariable<T>(other) {}

template <typename T>
placeholder<T>& placeholder<T>::operator = (const placeholder<T>& other)
{
	if (this == &other)
	{
		ileaf<T>::operator = (other);
	}
	return *this;
}

template <typename T>
placeholder<T>& placeholder<T>::operator = (placeholder<T>&& other)
{
	if (this == &other)
	{
		ileaf<T>::operator = (other);
	}
	return *this;
}

// maintains shape
template <typename T>
placeholder<T>& placeholder<T>::operator = (std::vector<T> data)
{
	// note: if this is allocated,
	// compatibility is compared to allocated shape instead of allowed
	assert(this->data_->is_compatible_with(data));

	if (false == this->data_->is_alloc())
	{
		if (optional<tensorshape> cand_shape = this->data_->guess_shape(data))
		{
			this->data_->allocate(*cand_shape);
		}
		else
		{
			throw std::exception(); // todo: better exception or warning + handling
		}
	}
	typename ileaf<T>::assignment* assigner =
		dynamic_cast<typename ileaf<T>::assignment*>(this->init_);
	(*assigner)(*this->data_, data);

	this->is_init_ = true;
	this->notify(react::UPDATE);
	return *this;
}

// changes shape
template <typename T>
placeholder<T>& placeholder<T>::operator = (tensor<T>& data)
{
	*this->data_ = std::move(data);
	this->is_init_ = true;
	this->notify(react::UPDATE);
	return *this;
}

template <typename T>
inode<T>* placeholder<T>::get_leaf (variable<T>* leaf)
{
	return this->zero.get();
}

template <typename T>
void placeholder<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const {}

template <typename T>
placeholder<T>::placeholder (const placeholder<T>& other) :
	ivariable<T>(other) {}

template <typename T>
inode<T>* placeholder<T>::clone_impl (void) const
{
	return new placeholder<T>(*this);
}

}

#endif