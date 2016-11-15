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
		constant (const constant<T>& other, std::string name);
		virtual ivariable<T>* clone_impl (std::string name);

	public:
		constant (T scalar);
		constant (std::vector<T> raw, tensorshape shape);

		// COPY
		constant<T>* clone (std::string name = "");

		// CONSTANT IS NOT A FIRST WORLD CITIZEN :(
		// override subject's detach method to suicide when lacking dependents
		virtual void detach (ccoms::iobserver* viewer);
};

}

#include "../../../src/graph/variable/constant.ipp"

#endif /* constant_hpp */