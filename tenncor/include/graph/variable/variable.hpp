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

// Never notifies... should consider inheriting from different parent
template <typename T>
class constant : public ileaf<T> {
	private:
		constant (const constant<T>& other, std::string name);

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		constant (T scalar);
		constant (std::vector<T> raw, tensor_shape shape);

		// COPY
		std::shared_ptr<constant<T> > clone (std::string name = "") {
			return std::static_pointer_cast<constant<T>, ievoker<T> >(clone_impl(name));
		}
		
		// CONSTANT IS NOT A FIRST WORLD CITIZEN :(
		virtual void detach (iobserver* viewer) {
			ccoms::subject::detach(viewer);
			if (this->no_audience()) {
				// no audience, no point to live x_x
				delete this;
			}
		}

		// FACTORIES
		static VAR_PTR<T> make (T scalar) {
			return ivariable<T>::make_shared(new constant(scalar));
		}
		static VAR_PTR<T> make (std::vector<T> raw, tensor_shape shape) {
			return ivariable<T>::make_shared(new constant(raw, shape));
		}
		static VAR_PTR<T> make (VAR_PTR<T> get_out) {
			return ivariable<T>::make_shared(new constant(get_out));
		}
};

// extend tensors by composition
// also holds initializer (in operation)f
template <typename T>
class variable : public ileaf<T> {
	private:
		variable (const variable<T>& other, std::string name);

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			// no need to set_gradient, gradient sets this as integral
			this->grad = safety_ref = ivariable<T>::gradient_leaf::make(this->self_ref_);
		}

	public:
		variable (T scalar);
		variable (std::string name = ""); // this requires initializer later on
		variable (const tensor_shape& shape, std::string name = "");
		variable (const tensor_shape& shape, initializer<T>& init, std::string name = "");

		// COPY
		std::shared_ptr<variable<T> > clone (std::string name = "") {
			return std::static_pointer_cast<variable<T>, ievoker<T> >(clone_impl(name));
		}

		// INITIALIZE VALUE
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensor_shape alloc_shape);

		// FACTORIES
		static VAR_PTR<T> make (T scalar) {
			return ivariable<T>::make_shared(new variable(scalar), true);
		}
		static VAR_PTR<T> make (std::string name = "") {
			return ivariable<T>::make_shared(new variable(name), true);
		}
		static VAR_PTR<T> make (const tensor_shape& shape, std::string name = "") {
			return ivariable<T>::make_shared(new variable(shape, name), true);
		}
		static VAR_PTR<T> make (const tensor_shape& shape, initializer<T>& init, std::string name = "") {
			return ivariable<T>::make_shared(new variable(shape, init, name), true);
		}
};

}

#include "../../../src/graph/variable/variable.ipp"

#endif /* variable_hpp */
