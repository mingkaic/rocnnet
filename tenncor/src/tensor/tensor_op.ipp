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
	info_ = other.info_;
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
	raw_update();
	return tensor<T,A>::get_raw();
}

template <typename T, typename A>
tensor_op<T,A>::tensor_op (TEN_OP<T> op, SHAPE shaper) : op_(op), shape_(shaper) {}

template <typename T, typename A>
tensor_op<T,A>::tensor_op (TEN_OP<T> op, SHAPE shaper, const alloc_attrib& attrib) :
	tensor<T,A>(std::vector<size_t>{}, attrib), op_(op), shape_(shaper) {}

template <typename T, typename A>
tensor_op<T,A>* tensor_op<T,A>::clone (void) { return static_cast<tensor_op<T,A>*>(clone_impl()); }

template <typename T, typename A>
tensor_op<T,A>& tensor_op<T,A>::operator = (tensor_op<T,A>& other)
{
	if (this != &other)
	{
		other.raw_update();
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
			// null are treated as 0
			raws_.push_back(&zero);
			shapes.push_back(std::vector<size_t>{1});
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
	info_.arg_shape_ = shapes;
	if (false == og_shape.is_fully_defined() ||
		false == og_shape.is_compatible_with(res_shape))
	{
		this->change_shape(res_shape);
		info_.res_shape_ = res_shape;
	}
	
	return *this;
}

template <typename T, typename A>
T tensor_op<T, A>::get (std::vector<size_t> indices)
{
	raw_update();
	return tensor<T, A>::get(indices);
}

}

#endif