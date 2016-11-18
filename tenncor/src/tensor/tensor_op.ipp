//
//  tensor_op.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-11.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef tensor_op_hpp

namespace nnet
{

template <typename T, typename A>
void tensor_op<T,A>::copy (const tensor_op<T,A>& other)
{
	op_ = other.op_;
	raws_ = other.raws_;
	tensor<T,A>::copy(other);
}

template <typename T, typename A>
tensor_op<T,A>::tensor_op (const tensor_op<T,A>& other)
{
	this->copy(other);
}

template <typename T, typename A>
tensor<T,A>* tensor_op<T,A>::clone_impl (void) { return new tensor_op<T,A>(*this); }

template <typename T, typename A>
T* tensor_op<T,A>::get_raw (void) {
	op_(this->get_raw(), raws_);
	return tensor<T,A>::get_raw();
}

template <typename T, typename A>
tensor_op<T,A>::tensor_op (TEN_OP<T> op) : op_(op) {}

template <typename T, typename A>
tensor_op<T,A>::tensor_op (TEN_OP<T> op, const alloc_attrib& attrib) :
	tensor<T,A>(std::vector<size_t>{}, attrib), op_(op) {}

template <typename T, typename A>
tensor_op<T,A>* tensor_op<T,A>::clone (void) { return static_cast<tensor_op<T,A>*>(clone_impl()); }

template <typename T, typename A>
tensor_op<T,A>& tensor_op<T,A>::operator = (const tensor_op<T,A>& other)
{
	if (this != &other)
	{
		this->copy(other);
	}
	return *this;
}

template <typename T, typename A>
const tensor_op<T,A>& tensor_op<T,A>::operator () (std::vector<tensor<T,A> const*> args)
{
	raws_.clear();
	for (tensor<T,A> const* t : args)
	{
		if (nullptr == t)
		{
			raws_.push_back(nullptr);
		}
		else
		{
			raws_.push_back(t->get_raw());
		}
	}
	return *this;
}

}

#endif