//
//  leaf.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_VARIABLE_HPP

namespace nnet
{

template <typename T>
variable<T>::variable (T scalar, std::string name) :
	ivariable<T>(std::vector<size_t>{1},
		new const_init<T>(scalar), name)
{
	initialize();
}

template <typename T>
variable<T>::variable (const tensorshape& shape, std::string name) :
	ivariable<T>(shape, nullptr, name) {}

template <typename T>
variable<T>::variable (const tensorshape& shape,
	const initializer<T>& init, std::string name) :
ivariable<T>(shape, init.clone(), name) {}

template <typename T>
variable<T>* variable<T>::clone (void) const
{
	return static_cast<variable<T>*>(clone_impl());
}

template <typename T>
variable<T>* variable<T>::move (void)
{
	return static_cast<variable<T>*>(move_impl());
}

template <typename T>
void variable<T>::set_initializer (const initializer<T>& init)
{
	if (this->init_)
	{
		delete this->init_;
	}
	this->init_ = init.clone();
}

template <typename T>
tensor<T>& variable<T>::initialize (void)
{
	assert(nullptr != this->init_);
	// if not alloc, attempt to allocate, throw if fail
	if (false == this->data_->is_alloc() &&
		false == this->data_->allocate())
	{
		throw std::runtime_error(this->get_label() + " data is not allocated");
	}
	initializer<T>* init = static_cast<initializer<T>*>(this->init_);
	(*init)(this->data_);
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensorshape shape)
{
	assert(this->init_ != nullptr);
	if (false == this->data_->allocate(shape))
	{
		std::stringstream ss;
		ss << "shape ";
		print_shape(shape, ss);
		ss << " failed to allocate " << this->get_label();
		throw std::runtime_error(ss.str());
	}
	initializer<T>* init = static_cast<initializer<T>*>(this->init_);
	(*init)(this->data_);
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

template <typename T>
void variable<T>::get_leaf (varptr<T>& out, variable<T>* leaf)
{
	if (this == leaf)
	{
		out = constant<T>::get_shared_one();
	}
	else
	{
		out = constant<T>::get_shared_zero();
	}
}

template <typename T>
void variable<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const
{
	leaves.emplace(const_cast<variable<T>*>(this), nullptr);
}

template <typename T>
variable_updater<T> variable<T>::assign (inode<T>* input) const
{
	return [this, input]()
	{
		tensor<T>* out_tens = this->data_;
		const tensor<T>* in_tens = input->get_eval();
		assign_func<T> assiger;
		assiger(out_tens, in_tens);
		this->notify(notification::UPDATE);
	};
}

template <typename T>
variable_updater<T> variable<T>::assign_add (inode<T>* input) const
{
	return [this, input]()
	{
		tensor<T>* out_tens = this->data_;
		const tensor<T>* in_tens = input->get_eval();
		assign_func<T> assign_add([](const T& e1, const T& e2) { return e1 + e2; });
		assign_add(out_tens, in_tens);
		this->notify(notification::UPDATE);
	};
}

template <typename T>
variable_updater<T> variable<T>::assign_sub (inode<T>* input) const
{
	return [this, input]()
	{
		tensor<T>* out_tens = this->data_;
		const tensor<T>* in_tens = input->get_eval();
		assign_func<T> assign_sub([](const T& e1, const T& e2) { return e1 - e2; });
		assign_sub(out_tens, in_tens);
		this->notify(notification::UPDATE);
	};
}

template <typename T>
inode<T>* variable<T>::clone_impl (void) const
{
	return new variable(*this);
}

template <typename T>
inode<T>* variable<T>::move_impl (void)
{
	return new variable(std::move(*this));
}

}

#endif
