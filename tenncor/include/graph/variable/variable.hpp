//
//  variable.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <list>
#include <ctime>
#include <random>
#include <new>
#include <memory>

#include "ileaf.hpp"
#include "constant.hpp"
#include "graph/state_selector/bindable_toggle.hpp"

#pragma once
#ifndef variable_hpp
#define variable_hpp

namespace nnet
{

// extend tensors by composition
// also holds initializer (in operation)

// TODO: merge variable and placeholder, add assignment node as a tensorless
// then move initializers as a updating delegate object (as oppose to the current variable delegate)
// variable may take assignments as an exclusive (non-reactive) dependency
template <typename T>
class variable : public ileaf<T>
{
	private:
		// >>>> GRAD INFO <<<<
		std::unique_ptr<bindable_toggle<T> > grad_ = nullptr; // make it variable to prevent self destruction when disconnecting

	protected:
		// avoid copying grad_
		variable (const variable<T>& other) : ileaf<T>(other) {}

		virtual void merge_leaves (std::unordered_set<ivariable<T>*>& src)
		{
			src.emplace(this);
		}

	public:
		variable (T scalar, std::string name = "scalar");
		variable (const tensorshape& shape, std::string name = "");
		variable (const tensorshape& shape, initializer<T>& init, std::string name = "");

		// COPY
		virtual variable<T>* clone (void);
		variable<T>& operator = (const variable<T>& other)
		{
			if (this != &other)
			{
				grad_.reset(nullptr);
				ileaf<T>::operator = (other);
			}
			return *this;
		}

		// INITIALIZE VALUE
		void set_initializer (initializer<T>& init);
		
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensorshape alloc_shape);

		// DATA EXPOSURE TO PARENT/DEPENDENT NODES
		virtual bindable_toggle<T>* get_gradient (void)
		{
			if (nullptr == this->grad_)
			{
				grad_ = std::unique_ptr<bindable_toggle<T> >(bindable_toggle<T>::build(
					constant<T>::build(0), constant<T>::build(1)));
			}
			return grad_.get();
		}
};

}

#include "../../../src/graph/variable/variable.ipp"

#endif /* variable_hpp */
