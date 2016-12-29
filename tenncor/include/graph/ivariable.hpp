//
//  ivariable.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "memory/session.hpp"
#include "initializer.hpp"
#include "tensor/tensor_op.hpp"
#include "tensor/tensor.hpp"
#include "graph/ccoms/subject_owner.hpp"
#include "utils/temp_utils.hpp"

#pragma once
#ifndef ivariable_hpp
#define ivariable_hpp

namespace nnet
{

template <typename T>
class assign;
template <typename T>
class ioptimizer;
template <typename T>
class iconnector;
template <typename T>
class functor;
template <typename T>
class bindable_toggle;

// VARIABLE INTERFACE

// DEFAULTS TO DOWN-UP VARIABLE (INFORMATION IS OBTAINED FROM LEAF NODES: Synthesized Attribute as oppose to Inherited)

// TODO: limit T down to numeric types using c++17 std::variant

template <typename T>
class ivariable : public ccoms::subject_owner
{
	private:
		// >>>> NODE IDS <<<<
		// unique hash
		const std::string id_ = r_temp::temp_uuid(this);
		// variable label
		std::string name_;

		template <typename U>
		friend ccoms::subject* var_to_sub (ivariable<U>* var);
		
	protected:
		// >>>> GRAPH INFO <<<<
		virtual void merge_leaves (std::unordered_set<ivariable<T>*>& src) = 0;

		// >>>> CONSTRUCTORS <<<<
		// copy constructor required to prevent id_ copyover
		ivariable (const ivariable<T>& other);
		// label constructor
		ivariable (std::string name);

		// protected members need to be accessed by other operations
//		friend class ioptimizer<T>;
		friend class iconnector<T>;
//		friend class functor<T>;

	public:
		virtual ~ivariable (void);
		
		// COPY
		// abstract clone
		// avoid copying id_
		ivariable<T>& operator = (const ivariable<T>& other);
		
		// MOVE
		// TODO Implement

		std::string get_uid (void) const { return id_; }
		virtual std::string get_name (void) const { return name_; }
		virtual tensorshape get_shape (void) = 0;
		
		// DATA EXPOSURE TO PARENT/DEPENDENT NODES
		// get eval simply returns the node's tensor
		// the node will not check if tensor is valid for evaluation...
		virtual tensor<T>* get_eval (void) = 0;
		virtual bindable_toggle<T>* get_gradient (void) = 0;
};

template <typename T>
ccoms::subject* var_to_sub (ivariable<T>* var)
{
	return var->caller_.get();
}

template <typename T>
ivariable<T>* sub_to_var (ccoms::subject* sub)
{
	return dynamic_cast<ivariable<T>*>(sub->get_owner());
}

template <typename T>
std::vector<T> expose (ivariable<T>* var)
{
	if (nullptr == var) return std::vector<T>{};
	tensor<T>* ten = var->get_eval();
	return expose(ten);
}

}

#include "../../src/graph/ivariable.ipp"

#endif /* ivariable_hpp */
