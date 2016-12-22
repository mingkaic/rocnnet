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
#include "ileaf.hpp"

#pragma once
#ifndef placeholder_hpp
#define placeholder_hpp

namespace nnet
{

template <typename T>
class placeholder : public ileaf<T>
{
	public:
		placeholder (const tensorshape& shape, std::string name = "");
		placeholder (const tensorshape& shape, initializer<T>& init, std::string name = "");

		// COPY
		virtual placeholder<T>* clone (void);

		// DATA ASSIGNMENT
		// assign raw data according to 1 dimension representation of inner tensor
		virtual placeholder<T>& operator = (std::vector<T> data);
		virtual placeholder<T>& operator = (tensor<T>& data);

		// MOVES
		// todo: implement move clone

		// DATA EXPOSURE TO PARENT/DEPENDENT NODES
		virtual ivariable<T>* get_gradient (void)
		{
			if (nullptr == this->grad_)
			{
				this->grad_ = std::make_unique<variable<T> >(0, "0");
			}
			return this->grad_.get();
		}
};

}

#include "../../../src/graph/variable/placeholder.ipp"

#endif /* placeholder_hpp */
