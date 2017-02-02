//
//  conditional.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-07.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "iselector.hpp"

#pragma once
#ifndef conditional_hpp
#define conditional_hpp

namespace nnet
{

template <typename T>
using PICK = std::function<size_t(std::vector<ivariable<T>*>)>;

// conditional selector

template <typename T>
class conditional : public iselector<T>
{
	private:
		PICK<T> picker_;

	protected:
		conditional (std::vector<ivariable<T>*> dependencies,
			PICK<T> picker, std::string name) :
			iselector<T>(dependencies, "conditional:"+name),
			picker_(picker) {}

	public:
		static conditional<T>* build (std::vector<ivariable<T>*> args,
			PICK<T> picker, std::string name = "")
		{
			return new conditional(args, picker, name);
		}

		virtual ~conditional (void) {}

		// COPY
		virtual conditional<T>* clone (void) { return new conditional<T>(*this); }

		// implement from ivariable
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			std::vector<ivariable<T>*> args =
				nnutils::to_vec<ccoms::subject*, ivariable<T>*>(this->dependencies_, sub_to_var<T>);
			this->active_ = picker_(args);
//			msg.grad_ = nullptr;
			this->notify(msg);
		}

		// gradient and jacobians can't be dynamically update as of yet
		virtual bindable_toggle<T>* get_gradient (void) { return nullptr; }
		virtual functor<T>* get_jacobian (void) { return nullptr; }
};

// select the variable that's not zero
// defaults to the first argument a
template <typename T>
varptr<T> not_zero (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == a) return b;
	else if (nullptr == b) return a;

	ivariable<T>* op = conditional<T>::build(std::vector<ivariable<T>*>{a, b},
	[](std::vector<ivariable<T>*> args)
	{
		ivariable<T>* a = args[0];
		tensor<T>* ta = a->get_eval();
		if (nullptr == ta) return 1;
		std::vector<T> ar = expose(ta);
		T tots = 0;
		for (T v : ar)
		{
			tots += v;
		}
		if (0 == tots)
		{
			return 1;
		}
		return 0;
	},
	"nonzero(" + a->get_name() + "," + b->get_name() + ")");
	return op;
}

}

#include "../../../src/graph/state_selector/conditional.ipp"

#endif /* conditional_hpp */
