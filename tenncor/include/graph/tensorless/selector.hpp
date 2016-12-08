//
//  selector.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-07.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/iconnector.hpp"

#pragma once
#ifndef selector_hpp
#define selector_hpp

namespace nnet
{

template <typename T>
using PICK = std::function<size_t(std::vector<ivariable<T>*>)>;

template <typename T>
class selector : public iconnector<T>
{
	private:
		size_t active_ = 0;
		PICK<T> picker_;

		ivariable<T>* get (size_t idx) { return sub_to_var<T>(this->dependencies_[idx]); }

	protected:
		selector (const selector<T>& other, std::string name) :
			iconnector<T>(other, name), active_(other.active_), picker_(other.picker_) {}

		virtual ivariable<T>* clone_impl (std::string name)
		{
			return new selector<T>(*this, name);
		}

		void copy (const selector<T>& other, std::string name = "")
		{
			active_ = other.active_;
			picker_ = other.picker_;
			iconnector<T>::copy(other, name);
		}

		selector (std::vector<ivariable<T>*> dependencies,
			PICK<T> picker, std::string name) :
			iconnector<T>(dependencies, name), picker_(picker) {}

	public:
		static selector<T>* build (std::vector<ivariable<T>*> args,
			PICK<T> picker, std::string name = "")
		{
			return new selector(args, picker, name);
		}

		virtual ~selector (void) {}

		// COPY
		selector<T>* clone (std::string name = "") { return static_cast<selector<T>*>(clone_impl(name)); }
		selector& operator = (const selector& other)
		{
			if (this != &other)
			{
				copy(other);
			}
			return *this;
		}

		// implement from ivariable
		virtual tensorshape get_shape (void) { return get(active_)->get_shape(); }
		virtual tensor<T>* get_eval (void) { return get(active_)->get_eval(); }

		// gradient and jacobians can't be dynamically update as of yet
		virtual ivariable<T>* get_gradient (void) { return nullptr; }
		virtual graph<T>* get_jacobian (void) { return nullptr; }

		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			std::vector<ivariable<T>*> args =
				nnutils::to_vec<ccoms::subject*, ivariable<T>*>(this->dependencies_, sub_to_var<T>);
			active_ = picker_(args);
			msg.grad_ = nullptr;
			this->notify(msg);
		}
};

// select the variable that's not zero
// defaults to the first argument a
template <typename T>
varptr<T> not_zero (const varptr<T> a, const varptr<T> b)
{
	if (nullptr == a) return b;
	ivariable<T>* op = selector<T>::build(std::vector<ivariable<T>*>{a, b},
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

#include "../../../src/graph/tensorless/selector.ipp"

#endif /* selector_hpp */
