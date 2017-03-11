//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-05.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TENSOR_HANDLER_HPP

namespace nnet
{

static std::default_random_engine generator(std::time(NULL));

template <typename T>
itensor_handler<T>::itensor_handler (SHAPER shaper, FORWARD_OP<T> forward) :
	shaper_(shaper), forward_(forward) {}

template <typename T>
void itensor_handler<T>::operator () (
	tensor<T>& out,
	std::vector<const tensor<T>&> args)
{
	std::vector<tensorshape> ts;
	std::vector<const T*> raws;
	for (const tensor<T>& arg : args)
	{
		ts.push_back(arg->get_shape());
		raws.push_back(arg.get_raw());
	}

	tensorshape s = shaper_(ts);
	tensorshape oshape = out.get_shape();
	if (false == s.is_compatible_with(oshape))
	{
		throw std::exception(); // TODO: better exception
	}
	forward_(out.get_raw(), oshape, raws);
}

template <typename T>
transfer_func<T>::transfer_func (SHAPER shaper, FORWARD_OP<T> forward) :
	itensor_handler<T>(shaper, forward) {}

template <typename T>
void transfer_func<T>::operator () (tensor<T>& out, std::vector<const tensor<T>&> args)
{
	transfer_func<T>::operator () (out, args);
}

template <typename T>
const_init<T>::const_init (T value) :
	itensor_handler<T>(
[](std::vector<tensorshape>) { return tensorshape(); },
[value](tensorshape shape,T* out,std::vector<const T*>)
{
	size_t len = shape.n_elems();
	std::fill(out, out+len-1, value);
}) {}

template <typename T>
void const_init<T>::operator () (tensor<T>& out)
{
	itensor_handler<T>::operator ()(out, {});
}

template <typename T>
rand_uniform<T>::rand_uniform (T min, T max) :
	distribution_(std::uniform_real_distribution<T>(min, max)),
	itensor_handler<T>(
[](std::vector<tensorshape>) { return tensorshape(); },
[this](tensorshape shape,T* out,std::vector<const T*>)
{
	size_t len = shape.n_elems();
	for (size_t i = 0; i < len; i++)
	{
		out[i] = distribution_(generator);
	}
}) {}

template <typename T>
void rand_uniform<T>::operator () (tensor<T>& out)
{
	itensor_handler<T>::operator ()(out, {});
}

}

#endif