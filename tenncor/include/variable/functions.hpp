//
//  functions.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef functions_ops
#define functions_ops

#include "unar_ops.hpp"
#include "bin_ops.hpp"

namespace nnet {

// FUNCTION WRAPPER

template <typename T>
class univar_func : public ioperation<T> {
	private:
		// do not own fanin or out
		ivariable<T>* fanin = nullptr;
		ioperation<T>* fanout = nullptr;
		// no longer need if use sharedptr

		void clear (void);
		void copy (const ivariable<T>& other, std::string name = "");

	protected:
		std::vector<ioperation<T>*> ownout;
		virtual void replace (
			const ivariable<T>& food,
			const ivariable<T>* newfood) {
			if (fanin == &food) fanin = const_cast<ivariable<T>*>(newfood);
		}

		virtual void shape_eval (void);
		univar_func (const univar_func<T>& other, std::string name);

	public:
		// declare
		univar_func (std::function<void(ioperation<T>*&)> declare);
		// currently shallow copy
		// TODO implement graph object manager for deep copy cloner
		virtual univar_func<T>* clone (std::string name = "");
		virtual ~univar_func (void) { clear(); }
		// connect input to fanin ivariables according
		// to declared equation ordered by function parameters
		virtual ivariable<T>& operator () (ivariable<T>& input);
		virtual univar_func<T>& operator = (const ivariable<T>& other);

		// calls derive from fanout
		virtual tensor<T>* derive (ivariable<T>* over) const;
		// calls derive on single input
		tensor<T>* derive (void) const;
		virtual const tensor<T>& eval (void);
};

// ACTIVATION FUNCTIONS

template <typename T>
class sigmoid : public univar_func<T> {
	public:
		sigmoid (void);
};

template <typename T>
class tanh : public univar_func<T> {
	public:
		tanh (void);
};

}

#include "../../src/variable/functions.tpp"

#endif /* functions_ops */
