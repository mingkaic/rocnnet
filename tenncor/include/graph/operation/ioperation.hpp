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

#include "executor/varptr.hpp"
#include "graph/buffer/igraph.hpp"

#pragma once
#ifndef ioperation_hpp
#define ioperation_hpp

namespace nnet
{

template <typename T>
using BUILD_DERIVE = std::function<ivariable<T>*(std::vector<ivariable<T>*>)>;

template <typename T>
class jacobian;
// matmul needs permission to change grad_jacobi_
template <typename T>
class matmul;

// INTERFACE OPERATION

template <typename T>
class ioperation : public iconnector<T>
{
	private:
		// buffer argument tensors 
		// (each argument can return different tensors)
		// update each tensor by position accordingly
		std::vector<tensor<T>*> tens_buffer_;

	protected:
		// WRAPPER CONTENT
		std::unique_ptr<tensor_op<T> > out_ = nullptr;

		bool valid_tensor_ = false;
		// shaper functional must return undefined shape in cases of error
		SHAPE shaper_;

		// GRADIENT CONTENTS
		iconnector<T>* grad_ = nullptr; // general gradient node
		igraph<T>* grad_jacobi_ = nullptr; // specific gradient node used for jacobians

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
		
		void setup_jacobian (igraph<T>* j)
		{
			if (nullptr == j) return;
			if (nullptr == grad_jacobi_)
			{
				// setup grad_jacobi_
				grad_jacobi_ = j;
				j->update_leaf([this](ivariable<T>* leaf, size_t idx)
				{
					// jacobian graph must have a singular leaf (the value of the buffer more specifically)
					assert(idx == 0);
					return this;
				});
			}
			// we can have at most 1 candiate grad_jacobi per node
			// we may have repeated candidates
			assert(grad_jacobi_ == j);
		}

		void message_update (ccoms::update_message msg)
		{
			// LEAF UPDATE
			if (msg.leave_update_)
			{
				this->leaves_update(); // TODO: have message notify whether or not to update leaves
			}
		}
		
		// CONSTRUCTS THE GRADIENT TREE AND STORE ROOT IN MEMBER GRAD
		virtual void setup_gradient (void) = 0; // ioperation specific

		void copy (const ioperation<T>& other, std::string name = "");
		virtual ivariable<T>* clone_impl (std::string name) = 0;
		ioperation (const ioperation<T>& other, std::string name);

		// used specifically to pass jacobian tensors up the tree... could make generic use in the future
		// combine with generalized notify/update
		virtual bool channel (std::stack<ivariable<T>*>& jacobi);

		ioperation (std::vector<ivariable<T>*> dependencies, std::string name);

		friend class gradient<T>;
		friend class jacobian<T>;
		friend class matmul<T>;

	public:
		virtual ~ioperation (void);

		// COPY
		ioperation<T>* clone (std::string name = "");
		virtual ioperation<T>& operator = (const ioperation<T>& other);
		
		// MOVE

		// implement from ivariable
		virtual tensorshape get_shape (void) const
		{
			if (false == valid_tensor_)
			{
				return tensorshape();
			}
			return this->out_->get_shape();
		}

		virtual tensor<T>* get_eval (void);

		virtual ivariable<T>* get_gradient (void); // access general gradient

		virtual igraph<T>* get_jacobian (void) { return grad_jacobi_; }
		
		// inherited by elementary and transform, overwritten by matmul and jacobian
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message());
};

}

#include "../../../src/graph/operation/ioperation.ipp"

#endif /* ioperation_hpp */
