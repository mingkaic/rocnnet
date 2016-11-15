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

template <typename T>
void tensor_op<T>::copy (const tensor_op<T>& other)
{
	op_ = other.op_;
	raws_ = other.raws_;
	tensor<T>::copy(other);
}

template <typename T>
tensor_op<T>::tensor_op (const tensor_op<T>& other)
{
	this->copy(other);
}

template <typename T>
tensor<T>* tensor_op<T>::clone_impl (void) { return new tensor_op<T>(*this); }

template <typename T>
T* tensor_op<T>::get_raw (void) {
	op_(this->get_raw(), raws_);
	return tensor<T>::get_raw();
}

template <typename T>
tensor_op<T>::tensor_op (TEN_OP<T> op) : op_(op) {}

template <typename T>
tensor_op<T>::tensor_op (TEN_OP<T> op, iallocator& alloc) :
	tensor<T>(std::vector<size_t>{}, alloc), op_(op) {}

template <typename T>
tensor_op<T>::tensor_op (TEN_OP<T> op, iallocator* alloc) :
	tensor<T>(std::vector<size_t>{}, alloc), op_(op) {}

template <typename T>
tensor_op<T>::tensor_op (TEN_OP<T> op, iallocator& alloc, const alloc_attrib& attrib) :
	tensor<T>(std::vector<size_t>{}, alloc, attrib), op_(op) {}

template <typename T>
tensor_op<T>::tensor_op (TEN_OP<T> op, iallocator* alloc, const alloc_attrib& attrib) :
	tensor<T>(std::vector<size_t>{}, alloc, attrib), op_(op) {}

template <typename T>
tensor_op<T>* tensor_op<T>::clone (void) { return static_cast<tensor_op<T>*>(clone_impl()); }

template <typename T>
tensor_op<T>& tensor_op<T>::operator = (const tensor_op<T>& other)
{
	if (this != &other)
	{
		this->copy(other);
	}
	return *this;
}

template <typename T>
const tensor_op<T>& tensor_op<T>::operator () (std::vector<tensor<T> const*> args)
{
	raws_.clear();
	for (tensor<T> const* t : args)
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