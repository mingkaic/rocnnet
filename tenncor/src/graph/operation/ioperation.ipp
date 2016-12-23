//
//  operation.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/variable/variable.hpp"

#ifdef ioperation_hpp

namespace nnet
{

// OPERATION INTERFACE UTILITY FUNCTIONS

template <typename T>
void ioperation<T>::copy (const ioperation<T>& other)
{
	// if grad is not being observed, then and only then delete
	if (nullptr != grad_ && grad_->no_audience())
	{
		grad_->safe_destroy();
	}
	// treat grad_jacobi_ exactly like grads
	if (nullptr != grad_jacobi_ && grad_jacobi_->no_audience())
	{
		grad_jacobi_->safe_destroy();
	}
	// shallow copy
	tens_buffer_ = other.tens_buffer_;
	
	// gradient data reset
	grad_ = nullptr;
	grad_jacobi_ = other.grad_jacobi_->clone();
}

template <typename T>
ioperation<T>::ioperation (const ioperation<T>& other) : iconnector<T>(other)
{
	this->copy(other);
}

template <typename T>
ioperation<T>::ioperation (std::vector<ivariable<T>*> dependencies, std::string name) :
	iconnector<T>(dependencies, name) {}

template <typename T>
ioperation<T>::~ioperation (void)
{
	if (nullptr != grad_jacobi_)
	{
		delete grad_jacobi_;
	}
}

template <typename T>
ioperation<T>& ioperation<T>::operator = (const ioperation<T>& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (other);
		this->copy(other);
	}
	return *this;
}

template <typename T>
tensor<T>* ioperation<T>::get_eval (void)
{
	if (false == valid_tensor_)
	{
		return nullptr;
	}
	return out_.get();
}

template <typename T>
ivariable<T>* ioperation<T>::get_gradient (void)
{
	if (nullptr == grad_)
	{
//		grad_ = std::unique_ptr<bindable_toggle<T> >(
//			bindable_toggle<T>::build(setup_gradient(), constant<T>::build(1)));
		grad_ = std::unique_ptr<iconnector<T> >(dynamic_cast<iconnector<T>*>(setup_gradient()));
		// set grad_ to null on safe_destroy
		grad_->set_death((void**) &grad_);
	}
	return grad_.get();
}

template <typename T>
void ioperation<T>::update (ccoms::caller_info info, ccoms::update_message msg)
{
	static tensor<T> ones(1);

	// UPDATING TENS_BUFFER
	// cast caller dependency as ivariable
	ivariable<T>* caller = info.caller_ ? sub_to_var<T>(info.caller_) : nullptr;
	ivariable<T>* grad = msg.grad_ ? sub_to_var<T>(msg.grad_) : nullptr;
	size_t callerid = info.caller_idx_;
	tensor<T>* storage = nullptr;
	this->valid_tensor_ = true;

	if (0 == tens_buffer_.size()) return;
	assert(callerid < this->dependencies_.size()); // same as caller is in dependencies
	// grab caller_id from message
	if (nullptr == grad) // don't care about grad, get best evaluation
	{
		// tensor buffer update
		storage = caller->get_eval();
		if (nullptr == storage ||
			false == storage->get_shape().is_fully_defined())
		{
			this->valid_tensor_ = false;
		}
	}
	else if (ileaf<T>* leaf = dynamic_cast<ileaf<T>*>(grad)) // eval if caller is grad, null otherwise
	{
		storage = leaf == caller ? leaf->get_eval() : nullptr;
	}
	else
	{
		storage = grad == caller ? &ones : nullptr;
	}
	// update caller tensor only
	tens_buffer_[callerid] = storage;

	// tensor update when ready
	if (this->valid_tensor_)
	{
		// null is treated as erroneous zero
		(*out_)(tens_buffer_);
	}
	msg.grad_ = nullptr;
	this->notify(msg);
}

}

#endif
