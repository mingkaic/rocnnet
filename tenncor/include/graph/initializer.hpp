//
//  initializer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <random>
#include "tensor/tensor.hpp"
#include "memory/session.hpp"

#pragma once
#ifndef initializer_hpp
#define initializer_hpp

namespace nnet
{
    
// INITIALIZERS

template <typename T>
class initializer
{
	protected:
	    // allow inheriteds to manipulate tensors (without knowledge of shape)
		void delegate_task (tensor<T>& ten, std::function<void(T*, size_t)> op);

	public:
		virtual ~initializer (void) {}
		virtual void operator () (tensor<T>& in) = 0;
		// initializers are far too simple to justify using clone_impl pattern
		// although smart pointers of initializers could be possible in the future
		virtual initializer<T>* clone (void) = 0;
};

template <typename T>
class const_init : public initializer<T>
{
	private:
		T value_;

	public:
		const_init (T value);
		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void);
};

template <typename T>
class random_uniform : public initializer<T>
{
	private:
		std::uniform_real_distribution<T>  distribution_;

	public:
		random_uniform (T min, T max);
		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void);
};
    
}

#include "../../src/graph/initializer.ipp"

#endif /* initializer_hpp */