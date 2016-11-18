//
//  constant.hpp
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
#ifndef constant_hpp
#define constant_hpp

namespace nnet
{

// Never notifies... should consider inheriting from different parent
template <typename T>
class constant : public ileaf<T>
{
	protected:
		// overriding subject: marks for self_destruction 
		// once subject detaches last observer
		virtual bool suicidal (void) { return true; }
		
		constant (const constant<T>& other, std::string name);
		virtual ivariable<T>* clone_impl (std::string name);

		constant (T scalar);
		constant (std::vector<T> raw, tensorshape shape);

	public:
		// build are necessary for suicidal leaves
		static constant<T>* build (T scalar)
		{
			return new constant<T>(scalar);
		}

		static constant<T>* build (std::vector<T> raw, tensorshape shape)
		{
			return new constant<T>(raw, shape);
		}

		// COPY
		constant<T>* clone (std::string name = "");
};

}

#include "../../../src/graph/variable/constant.ipp"

#endif /* constant_hpp */