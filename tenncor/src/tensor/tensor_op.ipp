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
	shape_ = other.shape_;
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
	if (false == this->is_alloc())
	{
		this->allocate();
	}
	T* dest = tensor<T,A>::get_raw();
	assert(false == raws_.empty());
	op_(this->alloc_shape_, dest, raws_);
	return dest;
}

template <typename T, typename A>
tensor_op<T,A>::tensor_op (TEN_OP<T> op, SHAPE shaper) : op_(op), shape_(shaper) {}

template <typename T, typename A>
tensor_op<T,A>::tensor_op (TEN_OP<T> op, SHAPE shaper, const alloc_attrib& attrib) :
	tensor<T,A>(std::vector<size_t>{}, attrib), op_(op), shape_(shaper) {}

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
const tensor_op<T,A>& tensor_op<T,A>::operator () (std::vector<tensor<T,A>*> args)
{
	raws_.clear();
	std::vector<tensorshape> shapes;
	for (tensor<T,A>* t : args)
	{
		if (nullptr == t)
		{
			raws_.push_back(nullptr);
		}
		else
		{
			raws_.push_back(t->get_raw());
			shapes.push_back(t->get_shape());
		}
	}
	// change shape?
	tensorshape og_shape = this->get_shape();
	tensorshape res_shape = shape_(shapes);
	if (false == og_shape.is_fully_defined() ||
		false == og_shape.is_compatible_with(res_shape))
	{
		this->change_shape(res_shape);
	}
	return *this;
}

}

#endif