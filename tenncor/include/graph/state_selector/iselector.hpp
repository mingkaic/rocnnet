//
//  iselector.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-07.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/iconnector.hpp"

#pragma once
#ifndef iselector_hpp
#define iselector_hpp

namespace nnet
{

template <typename T>
class iselector : public iconnector<T>
{
	private:
		ivariable<T>* get (size_t idx) { return sub_to_var<T>(this->dependencies_[idx]); }

	protected:
	    size_t active_ = 0; // all inherited control how this is set
	
		iselector (std::vector<ivariable<T>*> dependencies, std::string name) :
			iconnector<T>(dependencies, name) {}

	public:
		virtual ~iselector (void) {}

		// COPY
		// abstract clone

		// implement from ivariable
		virtual tensorshape get_shape (void) { return get(active_)->get_shape(); }
		virtual tensor<T>* get_eval (void) { return get(active_)->get_eval(); }

		// gradient and jacobians can't be dynamically update as of yet
		virtual ivariable<T>* get_gradient (void) { return nullptr; }
		virtual functor<T>* get_jacobian (void) { return nullptr; }

        // update remains abstract
};

}

#include "../../../src/graph/state_selector/iselector.ipp"

#endif /* iselector_hpp */
