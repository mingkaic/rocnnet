//
//  assign.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef assign_hpp

namespace nnet
{
	
template <typename T>
void assign<T>::copy (const assign<T>& other)
{
	dest_ = other.dest_;
	transfer_ = other.transfer_;
}
	
template <typename T>
assign<T>::assign (const assign<T>& other)
{
	copy(other);
}

template <typename T>
iexecutor<T>* assign<T>::clone_impl (void) 
{
	return new assign<T>(*this);
}

template <typename T>
assign<T>::assign (variable<T>* dest, ivariable<T>* src, ASSIGN_OP<T> trans) :
	transfer_(trans),
	dest_(dest)
{
	this->add(src);
}

template <typename T>
assign<T>* assign<T>::clone (void) 
{
	return static_cast<assign<T>*>(clone_impl());
}

template <typename T>
assign<T>& assign<T>::operator = (const assign<T>& other) 
{
	copy(other);
}
	
template <typename T>
void assign<T>::freeze (void)
{
	tensor<T>* out = dest_->get_eval();
	if (ivariable<T>* src = this->dependencies_[0])
	{
		tensor<T>* in = src->get_eval();
		if (out->is_same_size(*in))
		{
			local_cpy_.clear();
			const T* new_data = in->get_raw();
			std::copy(new_data, new_data+in->n_elems(), 
				std::back_inserter(local_cpy_));
		}
	}
	else
	{
		throw std::logic_error("assigning a non-variable node to a variable");
	}
}

template <typename T>
void assign<T>::execute (void)
{
	assert(false == local_cpy_.empty());
	tensor<T>* out = dest_->get_eval();
	T* old_data = out->get_raw();
	size_t total = local_cpy_.size();
	for (size_t i = 0; i < total; i++)
	{
		transfer_(old_data[i], local_cpy_[i]);
	}
	dest_->notify();
}

// assign sub

template <typename T>
assign_sub<T>::assign_sub (const assign_sub<T>& other) : 
	assign<T>(other) {}

template <typename T>
assign<T>* assign_sub<T>::clone_impl (void)
{
	return new assign_sub<T>(*this);
}

template <typename T>
assign_sub<T>::assign_sub (variable<T>* dest, ivariable<T>* src) :
	assign<T>(dest, src, [](T& target, T data) { target -= data; }) {}

template <typename T>
assign_sub<T>* assign_sub<T>::clone (void) 
{
	return static_cast<assign_sub<T>*>(clone_impl());
}

}

#endif