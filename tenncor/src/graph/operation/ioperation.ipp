//
//  operation.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef operation_hpp

namespace nnet
{

// OPERATION INTERFACE UTILITY FUNCTIONS

template <typename T>
void ioperation<T>::copy (const ioperation<T>& other, std::string name)
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
	grad_ = other.grad_;
	grad_jacobi_ = other.grad_jacobi_;
	tens_buffer_ = other.tens_buffer_;
	iconnector<T>::copy(other, name);
}

template <typename T>
ioperation<T>::ioperation (const ioperation<T>& other, std::string name) :
	iconnector<T>(other, name),
	valid_tensor_(other.valid_tensor_),
	grad_(other.grad_),
	grad_jacobi_(other.grad_jacobi_),
	tens_buffer_(other.tens_buffer_) {}
	
template <typename T>
bool ioperation<T>::channel (std::stack<ivariable<T>*>& jacobi)
{
	// propagate channel
	// did not implement jacobian conflicts resolution (when 2 jacobian nodes meeting at the same junction...)
	// as such, this is undefined behavior for now and should throw error
	size_t jacobi_count = 0;
	for (ccoms::subject* sub : this->dependencies_)
	{
		if (ivariable<T>* v = sub_to_var<T>(sub))
		{
			if (ioperation<T>* o = dynamic_cast<ioperation<T>*>(v)) {
				if (o->channel(jacobi)) {
					jacobi_count++;
				}
			}
		}
	}
	if (jacobi_count > 1)
	{
		throw std::logic_error("jacobian branch conflict occurred at " + this->get_name());
	}
	return jacobi_count != 0;
}

template <typename T>
ioperation<T>::ioperation (std::vector<ivariable<T>*> dependencies, std::string name) :
	iconnector<T>(dependencies, name)
{}

template <typename T>
ioperation<T>::~ioperation (void)
{
	if (nullptr != grad_)
	{
		delete grad_;
	}
	if (nullptr != grad_jacobi_)
	{
		delete grad_jacobi_;
	}
}

template <typename T>
ioperation<T>* ioperation<T>::clone (std::string name)
{
	return static_cast<ioperation<T>*>(clone_impl(name));
}

template <typename T>
ioperation<T>& ioperation<T>::operator = (const ioperation<T>& other)
{
	if (this != &other)
	{
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
		setup_gradient();
		// set grad_ to null on safe_destroy
		grad_->set_death((void**) &grad_);
	}
	return grad_;
}

template <typename T>
void ioperation<T>::update (ccoms::caller_info info, ccoms::update_message msg)
{
	static tensor<T> ones(1);

	// common update protocol based on the message
	message_update(msg);

	// UPDATING TENS_BUFFER
	// cast caller dependency as ivariable
	ivariable<T>* caller = info.caller_ ? sub_to_var<T>(info.caller_) : nullptr;
	ivariable<T>* grad = msg.grad_ ? sub_to_var<T>(msg.grad_) : nullptr;
	size_t callerid = info.caller_idx_;
	tensor<T>* storage = nullptr;
	this->valid_tensor_ = true;

	// if caller is null then update all tensors
	if (nullptr == caller)
	{
		// CALLER IS THIS. MEANING THIS IS CALLED DURING CONSTRUCTION
		tens_buffer_.clear();
		for (ccoms::subject* sub : this->dependencies_)
		{
			if (ivariable<T>* var = sub_to_var<T>(sub))
			{
				// GET JACOBIAN FROM CHILDREN DURING CONSTRUCTION
				if (ioperation<T>* op = dynamic_cast<ioperation<T>*>(var))
				{
					// grad_jacobi must be obtained from at most 1
					assert(nullptr == grad_jacobi_ ||
						   nullptr == op->grad_jacobi_);
					grad_jacobi_ = op->grad_jacobi_;
				}
				// tensor buffer initialize
				storage = var->get_eval();
				tens_buffer_.push_back(storage);
				if (nullptr == storage)
				{
					this->valid_tensor_ = false;
				}
			}
		}
	}
	else
	{
		if (0 == tens_buffer_.size()) return;
		assert(callerid < this->dependencies_.size()); // same as caller is in dependencies
		// grab caller_id from message
		if (nullptr == grad) // don't care about grad, get best evaluation
		{
			// tensor buffer update
			storage = caller->get_eval();
			if (nullptr == storage)
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
	}
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
