//
//  assign.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef assign_hpp
#define assign_hpp

#include "iexecutor.hpp"

namespace nnet {

template <typename T>
void direct (T& dest, T src) { dest = src; }

// NON-REACTIVE OPERATION

template <typename T>
class assign : public iexecutor<T> {
	private:
		// determines how element-wise assignment works, defaults to direct assignment
		std::function<void(T&,T)> transfer_;
		// target (weak pointer, no ownership)
		variable<T>* dest_ = nullptr;

	protected:
		assign (const assign<T>& other) {
			this->copy(other);
		}

		void copy (const assign<T>& other) {
			dest_ = other.dest_;
			transfer_ = other.transfer_;
			iexecutor<T>::copy(other);
		}

		virtual iexecutor<T>* clone_impl (void) { return new assign<T>(*this); }

	public:
		assign (variable<T>* dest, ivariable<T>* src, std::function<void(T&,T)> trans = direct);
		assign<T>* clone (void) { return static_cast<assign<T>*>(clone_impl()); }

		virtual void execute (void);
};

template <typename T>
class assign_sub : public assign<T> {
	protected:
		virtual assign<T>* clone_impl (void) {
			return new assign_sub<T>(*this);
		}

		assign_sub (const assign_sub<T>& other) : assign<T>(other) {}

	public:
		assign_sub (variable<T>* dest, ivariable<T>* src);

		assign_sub<T>* clone (void) { return static_cast<assign_sub<T>*>(clone_impl()); }
};

}

#include "../../../src/graph/bridge/assign.ipp"

#endif /* assign_hpp */
