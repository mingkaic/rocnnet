//
// Created by Mingkai Chen on 2016-11-06.
//

#pragma once
#ifndef inherited_hpp
#define inherited_hpp

#include "graph/ivariable.hpp"

namespace nnet {

// INTERFACE FOR INHERITED VARIABLES

// UP-DOWN VARIABLE (INFORMATION IS OBTAINED FROM SIBLINGS OR PARENTS: Inherited Attribute)

template <typename T>
class inherited_variable : public ivariable<T> {
	public:
		virtual ~inherited_variable (void) {}
		// eval remains abstract
		virtual ivariable<T>* get_gradient (void) = 0;
};

// all operations on jacobian will pass as operations on to_leaf_
// evaluation will call to_root_
template <typename T>
class jacobian : public inherited_variable<T> {
	protected:
		ivariable<T>* to_root_;
		ivariable<T>* to_leaf_;

		virtual void make_gradient (ivariable<T>*& safety_ref) {}
		virtual void set_gradient (ivariable<T>* g) {}

		// interaction control
		virtual void interact (ivariable<T>* op) {
			ivariable<T>* v = this->self_ref_.lock();
			op = &v;
		}
		virtual tensor<T>& grab_tensor (void) { return this->get_tensor_from(to_root_); }

		virtual ievoker<T>* clone_impl (std::string name) {
			return nullptr; // make deep copy later
		}

	public:
		jacobian (std::function<ivariable<T>*(ivariable<T>*)> construction) {
			to_root_ = new variable<T>(1);
			to_leaf_ = construction(to_root_);
			this->name = "J(" + to_leaf_->get_name() + ")";
		}

		ivariable<T>* clone (std::string name = "") {
			return std::static_pointer_cast<jacobian<T>, ievoker<T> >(clone_impl(name));
		}

		virtual const tensor<T>& eval (void) {
			return to_leaf_->eval();
		}

		// jacobian is inherently a gradient agent (it exists in some first order or above derivative function)
		// TODO will do nothing until second/nth order derivative is implemented
		virtual ivariable<T>* get_gradient (void) { return nullptr; }
};

}

#include "../../../src/graph/inherited/inherited.ipp"

#endif /* inherited_hpp */
