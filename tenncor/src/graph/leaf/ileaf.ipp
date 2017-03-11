//
//  ileaf.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_ILEAF_HPP

namespace nnet
{

template <typename T>
ileaf<T>::~ileaf (void)
{
	if (nullptr != init_)
	{
		delete init_;
	}
}

template <typename T>
ileaf<T>* ileaf<T>::clone (void) const
{
	return static_cast<ileaf<T>*>(this->clone_impl());
}

template <typename T>
ileaf<T>& ileaf<T>::operator = (const ileaf<T>& other)
{
	if (this != &other)
	{
		inode<T>::operator = (other);
		copy(other);
		this->notify(); // content changed
	}
	return *this;
}

template <typename T>
ileaf<T>& ileaf<T>::operator = (ileaf<T>&& other)
{
	if (this != &other)
	{
		inode<T>::operator = (other);
		move(other);
		this->notify(); // content changed
	}
}

template <typename T>
tensorshape ileaf<T>::get_shape (void) const
{
	if (nullptr != data_)
	{
		return data_->get_shape();
	}
	return std::vector<size_t>{};
}

template <typename T>
const tensor<T>* ileaf<T>::get_eval (void) const
{
	if (false == good_status())
	{
		return nullptr;
	}
	return data_.get();
}

template <typename T>
bool ileaf<T>::can_init (void) const
{
	return init_ != nullptr;
}

template <typename T>
bool ileaf<T>::good_status (void) const
{
	return is_init_;
}

template <typename T>
ileaf<T>::ileaf (const tensorshape& shape,
	itensor_handler<T>* init,
	std::string name) :
inode<T>(shape, name),
init_(init),
data_(new tensor<T>(shape)) {}

template <typename T>
ileaf<T>::ileaf (const ileaf<T>& other) :
	inode<T>(other)
{
	copy(other);
}

template <typename T>
ileaf<T>::ileaf (ileaf<T>&& other) :
	inode<T>(other)
{
	move(other);
}

template <typename T>
void ileaf<T>::copy (const ileaf<T>& other)
{
	if (nullptr != init_)
	{
		delete init_;
	}
	data_ = std::unique_ptr<tensor<T> >(other.data_->clone());
	init_ = other.init_->clone();
	is_init_ = other.is_init_;
}

template <typename T>
void ileaf<T>::move (ileaf<T>&& other)
{
	if (nullptr != init_)
	{
		delete init_;
	}
	data_ = std::unique_ptr<tensor<T> >(std::move(other.data_));
	init_ = std::move(other.init_);
	is_init_ = std::move(other.is_init_);
}

template <typename T>
class ileaf<T>::assignment : public itensor_handler<T>
{
public:
	assignment (void) :
		itensor_handler<T>(
	[](std::vector<tensorshape> ts) -> tensorshape
	{
		if (ts.empty()) return {};
		return ts[0];
	},
	[this](T* dest, tensorshape& shape, std::vector<const T*>& orig)
	{
		std::memcpy(dest, &temp_[0], shape.n_elems());
	}) {}

	// perform assignment
	void operator () (tensor<T>& out, std::vector<T>& data)
	{
		// update temp_
		temp_ = data;
		itensor_handler<T>::operator ()(out, {});
	}

protected:
	std::vector<size_t> temp_;
};

}

#endif