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

#pragma once
#ifndef variable_hpp
#define variable_hpp

#include "ileaf.hpp"

namespace nnet {

// extend tensors by composition
// also holds initializer (in operation)f
template <typename T>
class variable : public ileaf<T> {
	private:
		variable (const variable<T>& other, std::string name);

	protected:
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		variable (T scalar);
		variable (const tensor_shape& shape, std::string name = "");
		variable (const tensor_shape& shape, initializer<T>& init, std::string name = "");

		// COPY
        variable<T>* clone (std::string name = "") {
			return static_cast<variable<T>*>(clone_impl(name));
		}

		// INITIALIZE VALUE
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensor_shape alloc_shape);
};

}

#include "../../../src/graph/variable/variable.ipp"

#endif /* variable_hpp */
