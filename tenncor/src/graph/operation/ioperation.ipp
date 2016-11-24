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
	// shallow copy
	grad_ = other.grad_;
	ivariable<T>::copy(other, name);
}

template <typename T>
ioperation<T>::ioperation (const ioperation<T>& other, std::string name) :
	ccoms::iobserver(other),
	ivariable<T>(other, name),
	valid_tensor_(other.valid_tensor_),
	grad_(other.grad_) {}
	
template <typename T>
bool ioperation<T>::channel (std::stack<ivariable<T>*>& jacobi)
{
	// propagate channel
	// did not implement jacobian conflicts resolution (when 2 jacobian nodes meeting at the same junction...)
	// as such, this is undefined behavior for now and should throw error
	size_t jacobi_count = 0;
	for (ccoms::subject* sub : this->dependencies_)
	{
		if (ivariable<T>* v = dynamic_cast<ivariable<T>*>(sub))
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
	ccoms::iobserver(std::vector<ccoms::subject*>(dependencies.begin(), dependencies.end())),
	ivariable<T>(std::vector<size_t>{}, name) {}

template <typename T>
ioperation<T>::~ioperation (void)
{
	if (nullptr != grad_)
	{
		delete grad_;
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
	return this->out_.get();
}

template <typename T>
ivariable<T>* ioperation<T>::get_gradient (void)
{
	if (nullptr == grad_)
	{
		setup_gradient();
	}
	return grad_;
}

}

#endif
