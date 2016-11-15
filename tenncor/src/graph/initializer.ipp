//
//  initializer.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef initializer_hpp

namespace nnet
{

// INITIALIZERS

template <typename T>
void initializer<T>::delegate_task (tensor<T>& ten, std::function<void(T*, size_t)> op)
{
	op(ten.get_raw(), ten.n_elems());
}

template <typename T>
const_init<T>::const_init (T value) : 
    value_(value) {}

template <typename T>
void const_init<T>::operator () (tensor<T>& in)
{
	this->delegate_task(in, [this](T* raw, size_t len)
	{
		std::fill(raw, raw+len, value_);
	});
}

template <typename T>
initializer<T>* const_init<T>::clone (void)
{
	return new const_init(value_);
}

template <typename T>
random_uniform<T>::random_uniform (T min, T max)
{
	distribution_ = std::uniform_real_distribution<T>(min, max);
}

template <typename T>
void random_uniform<T>::operator () (tensor<T>& in)
{
	this->delegate_task(in, [this](T* raw, size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			raw[i] =  distribution_(session::get_generator());
		}
	});
}

template <typename T>
initializer<T>* random_uniform<T>::clone (void)
{
	return new random_uniform<T>(distribution_.min(), distribution_.max());
}

}

#endif