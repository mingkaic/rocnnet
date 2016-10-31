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

#include "ivariable.hpp"

namespace nnet {

template <typename T>
class constant : public ivar_init<T> {
	private:
		constant (T scalar);
		constant (std::vector<T> raw, tensor_shape shape);
		constant (VAR_PTR<T> get_out);

	protected:
		constant (const constant<T>& other, std::string name);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			VAR_PTR<T> g = std::shared_ptr<constant<T> >(new constant(0));
			this->set_gradient(g);
			safety_ref = g;
		}

	public:
		static VAR_PTR<T> make (T scalar) {
			return ivariable<T>::make_shared(new constant(scalar));
		}
		static VAR_PTR<T> make (std::vector<T> raw, tensor_shape shape) {
			return ivariable<T>::make_shared(new constant(raw, shape));
		}
		static VAR_PTR<T> make (VAR_PTR<T> get_out) {
			return ivariable<T>::make_shared(new constant(get_out));
		}
		virtual ~constant (void) {}

		std::shared_ptr<constant<T> > clone (std::string name = "") {
			return std::static_pointer_cast<constant<T>, ievoker<T> >(clone_impl(name));
		}
};

// extend tensors by composition
// also holds initializer (in operation)f
template <typename T>
class variable : public ivar_init<T> {
	private:
		variable (T scalar);
		variable (std::string name = ""); // this requires initializer later on
		variable (const tensor_shape& shape, std::string name = "");
		variable (const tensor_shape& shape, initializer<T>& init, std::string name = "");

	protected:
		variable (const variable<T>& other, std::string name);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			// no need to set_gradient, gradient sets this as integral
			this->grad = safety_ref = ivariable<T>::gradient_leaf::make(this-> self_ref_);
		}

	public:
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
		virtual ~variable (void) {}

		std::shared_ptr<variable<T> > clone (std::string name = "") {
			return std::static_pointer_cast<variable<T>, ievoker<T> >(clone_impl(name));
		}

		// required by variables using initializer (not by placeholder)
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensor_shape alloc_shape);
};

template <typename T>
class placeholder : public ivar_init<T> {
	private:
		void consumer_reshape (void);
		placeholder (const placeholder<T>& other, std::string name);
		placeholder (std::string name); // super generic placeholder
		placeholder (const tensor_shape& shape, std::string name);

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static std::shared_ptr<placeholder<T> > make (std::string name = "") {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new placeholder(name));
			return std::static_pointer_cast<placeholder<T> >(inst);
		}
		static std::shared_ptr<placeholder<T> > make (const tensor_shape& shape, std::string name = "") {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new placeholder(shape, name));
			return std::static_pointer_cast<placeholder<T> >(inst);
		}
		virtual ~placeholder (void) {}

		std::shared_ptr<placeholder<T> > clone (std::string name = "") {
			return std::static_pointer_cast<placeholder<T>, ievoker<T> >(clone_impl(name));
		}

		// assign raw data according to 1 dimension representation of inner tensor
		virtual ivariable<T>& assign (VAR_PTR<T> other);
		virtual ivariable<T>& operator = (std::vector<T> data);
		virtual ivariable<T>& operator = (const tensor<T>& data);

		// replace with shared_ptr<unique_ptr<placeholder<T> > >...
		void replace (const placeholder<T>& other);
};

template <typename T>
using PLACEHOLDER_PTR = std::shared_ptr<placeholder<T> >;

template <typename T>
constexpr auto PLACEHOLDER_TO_VAR = std::static_pointer_cast<ivariable<T>, placeholder<T> >;

}

#include "../../../src/graph/variable/variable.ipp"

#endif /* variable_hpp */
