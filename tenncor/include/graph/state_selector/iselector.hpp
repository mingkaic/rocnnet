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

		// by default, we should grab the active state's gradient, and jacobian
		// but know that gradient and jacobian nodes will NEVER be reassigned as of the current implementation
		// TODO: enable gradient and jacoabian node update when selectors change states (too much work?)
		virtual bindable_toggle<T>* get_gradient (void) { return get(active_)->get_gradient(); }
		virtual functor<T>* get_jacobian (void)
		{
			if (iconnector<T>* c = dynamic_cast<iconnector<T>*>(get(active_)))
			{
				return c->get_jacobian();
			}
			return nullptr;
		}

        // update remains abstract
};

}

#include "../../../src/graph/state_selector/iselector.ipp"

#endif /* iselector_hpp */
