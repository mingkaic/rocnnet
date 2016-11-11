//
//  placeholder.hpp
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
#ifndef placeholder_hpp
#define placeholder_hpp

#include "ileaf.hpp"

namespace nnet {

template <typename T>
class placeholder : public ileaf<T> {
	private:
		placeholder (const placeholder<T>& other, std::string name);

	protected:
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		placeholder (const tensor_shape& shape, std::string name = "");
		placeholder (const tensor_shape& shape, initializer<T>& init, std::string name = "");
			
		// COPY
        placeholder<T>* clone (std::string name = "") {
			return static_cast<placeholder<T>*>(clone_impl(name));
		}

		// DATA ASSIGNMENT
		// assign raw data according to 1 dimension representation of inner tensor
		virtual placeholder<T>& operator = (std::vector<T> data);
		virtual placeholder<T>& operator = (const tensor<T>& data);

		// MOVES
		// todo: implement move clone
};

}

#include "../../../src/graph/variable/placeholder.ipp"

#endif /* placeholder_hpp */
