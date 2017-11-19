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
	delete data_;
}

template <typename T>
ileaf<T>* ileaf<T>::clone (void) const
{
	return static_cast<ileaf<T>*>(this->clone_impl());
}

template <typename T>
ileaf<T>* ileaf<T>::move (void)
{
	return static_cast<ileaf<T>*>(this->move_impl());
}

template <typename T>
ileaf<T>& ileaf<T>::operator = (const ileaf<T>& other)
{
	if (this != &other)
	{
		inode<T>::operator = (other);
		copy_helper(other);
		this->notify(UPDATE); // content changed
	}
	return *this;
}

template <typename T>
ileaf<T>& ileaf<T>::operator = (ileaf<T>&& other)
{
	if (this != &other)
	{
		inode<T>::operator = (other);
		move_helper(std::move(other));
		this->notify(UPDATE); // content changed
	}
	return *this;
}

template <typename T>
size_t ileaf<T>::get_depth (void) const
{
	return 0; // leaves are 0 distance from the furthest dependent leaf
}

template <typename T>
std::vector<inode<T>*> ileaf<T>::get_arguments (void) const
{
	return {};
}

template <typename T>
size_t ileaf<T>::n_arguments (void) const
{
	return 0;
}

template <typename T>
const tensor<T>* ileaf<T>::eval (void)
{
	return get_eval();
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
std::unordered_set<ileaf<T>*> ileaf<T>::get_leaves (void) const
{
	return {const_cast<ileaf<T>*>(this)};
}

template <typename T>
bool ileaf<T>::good_status (void) const
{
	return is_init_;
}

template <typename T>
bool ileaf<T>::read_proto (const tenncor::tensor_proto& proto)
{
	bool success = data_->from_proto(proto);
	if (success)
	{
		is_init_ = true;
		this->notify(UPDATE);
	}
	return success;
}

template <typename T>
ileaf<T>::ileaf (const tensorshape& shape, std::string name) :
	inode<T>(name),
	data_(new tensor<T>(shape)) {}

template <typename T>
ileaf<T>::ileaf (const ileaf<T>& other) :
	inode<T>(other)
{
	copy_helper(other);
}

template <typename T>
ileaf<T>::ileaf (ileaf<T>&& other) :
	inode<T>(std::move(other))
{
	move_helper(std::move(other));
}

template <typename T>
const tensor<T>* ileaf<T>::get_eval (void) const
{
	if (false == good_status())
	{
		return nullptr;
	}
	return data_;
}

template <typename T>
void ileaf<T>::copy_helper (const ileaf<T>& other)
{
	if (data_)
	{
		delete data_;
		data_ = nullptr;
	}
	is_init_ = other.is_init_;
	// copy over data if other has good_status (we want to ignore uninitialized data)
	if (other.data_)
	{
		data_ = other.data_->clone(!other.good_status());
	}
}

template <typename T>
void ileaf<T>::move_helper (ileaf<T>&& other)
{
	if (data_)
	{
		delete data_;
	}
	is_init_ = std::move(other.is_init_);
	data_ = std::move(other.data_);
	other.data_ = nullptr;
}

}

#endif