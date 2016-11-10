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
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		constant (T scalar);
		constant (std::vector<T> raw, tensor_shape shape);

		// COPY
        constant<T>* clone (std::string name = "") {
			return static_cast<constant<T>*>(clone_impl(name));
		}

		// CONSTANT IS NOT A FIRST WORLD CITIZEN :(
		virtual void detach (ccoms::iobserver* viewer) {
			ccoms::subject::detach(viewer);
			if (this->no_audience()) {
				// no audience, no point to live x_x
				delete this;
			}
		}

		// FACTORIES
		static ivariable<T>* make (T scalar) {
			return ivariable<T>::make_shared(new constant(scalar));
		}
		static ivariable<T>* make (std::vector<T> raw, tensor_shape shape) {
			return ivariable<T>::make_shared(new constant(raw, shape));
		}
		static ivariable<T>* make (ivariable<T>* get_out) {
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
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		variable (T scalar);
		variable (std::string name = ""); // this requires initializer later on
		variable (const tensor_shape& shape, std::string name = "");
		variable (const tensor_shape& shape, initializer<T>& init, std::string name = "");

		// COPY
        variable<T>* clone (std::string name = "") {
			return static_cast<variable<T>*>(clone_impl(name));
		}

		variable<T>& operator = (const ivariable<T>*& other) {
			if (this != other) {
				this->copy(*other);
			}
			return *this;
		}

		// INITIALIZE VALUE
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensor_shape alloc_shape);

		// FACTORIES
		static ivariable<T>* make (T scalar) {
			return ivariable<T>::make_shared(new variable(scalar), true);
		}
		static ivariable<T>* make (std::string name = "") {
			return ivariable<T>::make_shared(new variable(name), true);
		}
		static ivariable<T>* make (const tensor_shape& shape, std::string name = "") {
			return ivariable<T>::make_shared(new variable(shape, name), true);
		}
		static ivariable<T>* make (const tensor_shape& shape, initializer<T>& init, std::string name = "") {
			return ivariable<T>::make_shared(new variable(shape, init, name), true);
		}
};

}

#include "../../../src/graph/variable/variable.ipp"

#endif /* variable_hpp */
