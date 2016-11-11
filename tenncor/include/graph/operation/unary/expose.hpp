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

#include "iunar_ops.hpp"

namespace nnet {

// OUT NODE

template <typename T>
class expose : public iunar_ops<T> {
	private:
		expose (expose<T>& var, std::string name) { ioperation<T>::copy(var, name); }
		
	protected:
		// backward chaining for AD
		virtual void setup_gradient (void) {}
		virtual ievoker<T>* clone_impl (std::string name);

		std::vector<T> get_vec (const tensor<T>& in) const;
		std::string get_symb (void) { return "expose"; }

	public:
		expose (ivariable<T>* var) { this->init(var); }

		// COPY
        expose<T>* clone (std::string name = "") {
			return static_cast<expose<T>*>(clone_impl(name));
		}
		
		// MOVES
		// TODO: implement

		virtual void update (void);

		// non-inheriteds
		// evaluates consumed operation
		virtual std::vector<T> get_raw (void);
};

}

#include "../../../../src/graph/operation/unary/expose.ipp"

#endif /* expose_hpp */
