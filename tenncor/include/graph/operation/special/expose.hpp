//
//  expose.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef expose_hpp
#define expose_hpp

#include "../ioperation.hpp"

namespace nnet {

// OUT NODE

template <typename T>
class expose : public ioperation<T> {
	protected:
		// backward chaining for AD
		virtual tensorshape shape_eval (void);
		virtual void setup_gradient (void) {}
		virtual ivariable<T>* clone_impl (std::string name);

		expose (const expose<T>& other, std::string name) :
			ioperation<T>(other, name) {}

	public:
		expose (ivariable<T>* var) : ioperation<T>(
			std::vector<ivariable<T>*>{var}, "<expose>(" + var->get_name() + ")") {}

		// COPY
		expose<T>* clone (std::string name = "") {
			return static_cast<expose<T>*>(clone_impl(name));
		}
		
		// MOVES
		// TODO: implement

		virtual void update (ccoms::subject* caller);

		// non-inherited
		virtual std::vector<T> get_raw (void);
};

}

#include "../../../../src/graph/operation/special/expose.ipp"

#endif /* expose_hpp */
