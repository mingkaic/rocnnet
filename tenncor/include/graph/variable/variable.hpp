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

#pragma once
#ifndef variable_hpp
#define variable_hpp

namespace nnet
{

// extend tensors by composition
// also holds initializer (in operation)f
template <typename T>
class variable : public ileaf<T>
{
	protected:
		variable (const variable<T>& other, std::string name);
		virtual ivariable<T>* clone_impl (std::string name);

	public:
		variable (T scalar);
		variable (const tensorshape& shape, std::string name = "");
		variable (const tensorshape& shape, initializer<T>& init, std::string name = "");

		// COPY
		variable<T>* clone (std::string name = "");

		// INITIALIZE VALUE
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensorshape alloc_shape);
};

}

#include "../../../src/graph/variable/variable.ipp"

#endif /* variable_hpp */
