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
		this->notify(react::UPDATE); // content changed
	}
	return *this;
}

template <typename T>
ileaf<T>& ileaf<T>::operator = (ileaf<T>&& other)
{
	if (this != &other)
	{
		inode<T>::operator = (other);
		move(std::move(other));
		this->notify(react::UPDATE); // content changed
	}
	return *this;
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
bool ileaf<T>::good_status (void) const
{
	return is_init_;
}

template <typename T>
ileaf<T>::ileaf (const tensorshape& shape, std::string name) :
	inode<T>(name),
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
	move(std::move(other));
}

template <typename T>
void ileaf<T>::copy (const ileaf<T>& other)
{
	data_ = std::unique_ptr<tensor<T> >(other.data_->clone());
	is_init_ = other.is_init_;
}

template <typename T>
void ileaf<T>::move (ileaf<T>&& other)
{
	data_ = std::unique_ptr<tensor<T> >(std::move(other.data_));
	is_init_ = std::move(other.is_init_);
}

template <typename T>
class ileaf<T>::assignment : public initializer<T>
{
public:
	assignment (void) :
		initializer<T>(
	[](std::vector<tensorshape> ts) -> tensorshape
	{
		if (ts.empty()) return {};
		return ts[0];
	},
	[this](T* dest, const tensorshape& shape, std::vector<const T*>&,std::vector<tensorshape>&)
	{
		std::memcpy(dest, &temp_[0], shape.n_elems());
	}) {}

	assignment* clone (void) const
	{
		return static_cast<assignment*>(clone_impl());
	}

	// perform assignment
	void operator () (tensor<T>& out, std::vector<T>& data)
	{
		// update temp_
		temp_ = data;
		itensor_handler<T>::operator ()(out, {});
	}

protected:
	assignment (const assignment& other) : initializer<T>(other) {}

	virtual itensor_handler<T>* clone_impl (void) const
	{
		return new assignment(*this);
	}

	std::vector<T> temp_;
};

}

#endif