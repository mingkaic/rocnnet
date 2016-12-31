//
//  operation.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <unordered_map>
#include <algorithm>
#include <random>
#include <functional>
#include <limits>
#include <memory>
#include <stack>

#include "graph/variable/constant.hpp"
#include "graph/tensorless/functor.hpp"
#include "graph/state_selector/bindable_toggle.hpp"

#pragma once
#ifndef ioperation_hpp
#define ioperation_hpp

namespace nnet
{

template <typename T>
using BUILD_DERIVE = std::function<ivariable<T>*(std::vector<ivariable<T>*>)>;

// INTERFACE OPERATION

template <typename T>
class ioperation : public iconnector<T>
{
	private:
		// buffer argument tensors 
		// (each argument can return different tensors)
		// update each tensor by position accordingly
		std::vector<tensor<T>*> tens_buffer_;
		
		void copy (const ioperation<T>& other);

	protected:
		// WRAPPER CONTENT
		std::unique_ptr<tensor_op<T> > out_ = nullptr;

		bool valid_tensor_ = false;
		// shaper functional must return undefined shape in cases of error
		SHAPE shaper_;

		// >>>> GRAD INFO <<<<
		std::unique_ptr<bindable_toggle<T> > grad_ = nullptr; // general gradient node
		functor<T>* grad_jacobi_ = nullptr; // specific gradient node used for jacobians

		ioperation (const ioperation<T>& other);

		// to extract shape info
		// this shape evaluation is for when arguments are not instantiated
		// uninstantiated arguments evade the shape_eval phase
		// naturally, shaper_ is passed into tensor_op which would evaluate shape at update time
		virtual tensorshape shape_eval (void)
		{
			std::vector<tensorshape> shapes;
			for (ccoms::subject* sub : this->dependencies_)
			{
				if (ivariable<T>* v = sub_to_var<T>(sub))
				{
					shapes.push_back(v->get_shape());
				}
			}
			return shaper_(shapes);
		}
		
		void setup_jacobian (functor<T>* j)
		{
			if (nullptr == j) return;
			if (nullptr == grad_jacobi_)
			{
				functor<T>* my_jacobi = j->append_leaf(this);
				// set my_jacobi as grad_jacobi_
				set_jacobian(my_jacobi);
			}
			else
			{
				throw std::exception(); // we can only successfully setup once
			}
		}
		
		// CONSTRUCTS THE GRADIENT TREE AND STORE ROOT IN MEMBER GRAD
		virtual ivariable<T>* setup_gradient (void) = 0; // ioperation specific

		// set up tens_buffer
		void initialize (void)
		{
			this->valid_tensor_ = true;
			tens_buffer_.clear();
			for (ccoms::subject* sub : this->dependencies_)
			{
				if (ivariable<T>* var = sub_to_var<T>(sub))
				{
					// grab jacobian
					if (iconnector<T>* c = dynamic_cast<iconnector<T>*>(var))
					{
						// if we get arguments with jacobians, we are most likely in reverse mode graph
						setup_jacobian(c->get_jacobian());
					}
					// tensor buffer initialize
					tensor<T>* temp = var->get_eval();
					tens_buffer_.push_back(temp);
					if (nullptr == temp || false == temp->get_shape().is_fully_defined())
					{
						this->valid_tensor_ = false;
					}
				}
			}
			if (this->valid_tensor_)
			{
				// null is treated as erroneous zero
				(*out_)(tens_buffer_);
			}
		}

		ioperation (std::vector<ivariable<T>*> dependencies, std::string name);

		friend class gradient<T>;

	public:
		virtual ~ioperation (void);

		// COPY
		virtual ioperation<T>& operator = (const ioperation<T>& other);
		
		// MOVE

		// implement from ivariable
		virtual tensorshape get_shape (void)
		{
			if (false == valid_tensor_)
			{
				return tensorshape();
			}
			return this->out_->get_shape();
		}

		virtual tensor<T>* get_eval (void);

		virtual bindable_toggle<T>* get_gradient (void); // access general gradient

		void set_jacobian (functor<T>* j)
		{
			if (nullptr == grad_jacobi_)
			{
				grad_jacobi_ = j;
				grad_jacobi_->set_death((void**) &grad_jacobi_);
			}
		}
		virtual functor<T>* get_jacobian (void) { return grad_jacobi_; }
		
		// inherited by elementary and transform, overwritten by matmul and jacobian
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message());
};

}

#include "../../../src/graph/operation/ioperation.ipp"

#endif /* ioperation_hpp */
