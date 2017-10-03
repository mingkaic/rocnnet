//
//  generator.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-18.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#ifdef ROCNNET_GENERATOR_HPP

namespace nnet
{

template <typename T>
generator<T>::~generator (void)
{
	clean_up();
}

template <typename T>
generator<T>* generator<T>::get (inode<T>* shape_dep,
	const initializer<T>& init, std::string name)
{
	return new generator(shape_dep, init, name);
}

template <typename T>
generator<T>* generator<T>::clone (void) const
{
	return static_cast<generator<T>*>(this->clone_impl());
}

template <typename T>
generator<T>* generator<T>::move (void)
{
	return static_cast<generator<T>*>(this->move_impl());
}

template <typename T>
generator<T>& generator<T>::operator = (const generator<T>& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (other);
		clean_up();
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

template <typename T>
generator<T>& generator<T>::operator = (generator<T>&& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (std::move(other));
		clean_up();
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}

template <typename T>
void generator<T>::temporary_eval (const iconnector<T>*, inode<T>*& out) const
{
	out = constant<T>::get(1);
}

template <typename T>
varptr<T> generator<T>::derive (inode<T>* wrt)
{
	if (this != wrt)
	{
		return constant<T>::get_shared_zero();
	}
	return constant<T>::get_shared_one();
}

template <typename T>
tensorshape generator<T>::get_shape (void) const
{
	if (nullptr == data_)
	{
		return tensorshape{};
	}
	return data_->get_shape();
}

template <typename T>
std::unordered_set<ileaf<T>*> generator<T>::get_leaves (void) const
{
	return std::unordered_set<ileaf<T>*>{};
}

template <typename T>
bool generator<T>::good_status (void) const
{
	return nullptr != data_;
}

template <typename T>
bool generator<T>::read_proto (const tenncor::tensor_proto&) { return false; }

template <typename T>
void generator<T>::update (std::unordered_set<size_t>)
{
	inode<T>* dep = dynamic_cast<inode<T>*>(this->dependencies_[0]);
	if (nullptr == dep)
	{
		// self destroy
		this->notify(UNSUBSCRIBE);
	}
	tensorshape depshape = dep->get_shape();
	if (false == dep->good_status() || false == depshape.is_fully_defined())
	{
		return;
	}
	if (nullptr == data_)
	{
		// init
		data_ = new tensor<T>(depshape);
		(*init_)(*data_);
		this->notify(UPDATE);
	}
	else if (false == data_->get_shape().is_compatible_with(depshape))
	{
		// reshape
		data_->set_shape(depshape);
		(*init_)(*data_);
		this->notify(UPDATE);
	}
	else
	{
		// change shape
	}
}

template <typename T>
generator<T>::generator (inode<T>* shape_dep, const initializer<T>& init, std::string name) :
	iconnector<T>({shape_dep}, name)
{
	this->init_ = init.clone();
	this->update(std::unordered_set<size_t>{});
}

template <typename T>
generator<T>::generator (const generator<T>& other) :
	iconnector<T>(other)
{
	copy_helper(other);
}

template <typename T>
generator<T>::generator (generator<T>&& other) :
	iconnector<T>(std::move(other))
{
	move_helper(std::move(other));
}

template <typename T>
inode<T>* generator<T>::clone_impl (void) const
{
	return new generator(*this);
}

template <typename T>
inode<T>* generator<T>::move_impl (void)
{
	return new generator(std::move(*this));
}

template <typename T>
const tensor<T>* generator<T>::get_eval (void) const
{
	return data_;
}

template <typename T>
inode<T>* generator<T>::get_gradient (variable<T>*)
{
	return constant<T>::get_shared_zero();
}

template <typename T>
void generator<T>::death_on_broken (void)
{
	delete this;
}

template <typename T>
void generator<T>::death_on_noparent (void)
{
	delete this;
}

template <typename T>
void generator<T>::copy_helper (const generator<T>& other)
{
	if (data_)
	{
		delete data_;
	}
	if (init_)
	{
		delete init_;
	}

	if (other.init_)
	{
		init_ = other.init_->clone();
	}
	if (other.data_)
	{
		data_ = other.data_->clone();
	}
}

template <typename T>
void generator<T>::move_helper (generator<T>&& other)
{
	if (data_)
	{
		delete data_;
	}
	if (init_)
	{
		delete init_;
	}

	if (other.init_)
	{
		init_ = other.init_->move();
	}
	if (other.data_)
	{
		data_ = other.data_->move();
	}
}

template <typename T>
void generator<T>::clean_up (void)
{
	if (init_) delete init_;
	if (data_) delete data_;
	init_ = nullptr;
	data_ = nullptr;
}

}

#endif
