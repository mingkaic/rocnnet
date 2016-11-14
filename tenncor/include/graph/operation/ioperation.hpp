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

#pragma once
#ifndef operation_hpp
#define operation_hpp

#include "../variable/variable.hpp"
#include "graph/ccoms/iobserver.hpp"
#include "tensor/tensor_jacobi.hpp"

namespace nnet {

template <typename T>
class gradient;

template <typename T>
using BUILD_DERIVE = std::function<ivariable<T>*(std::vector<ivariable<T>*>)>;

// INTERFACE OPERATION

// inheritance join at ileaf_handler
template <typename T>
class ioperation : virtual public ivariable<T>, virtual public ccoms::iobserver {
	protected:
		bool valid_tensor_ = false;
		ivariable<T>* grad_ = nullptr;

		// implement unique method of consuming input variables
		// to extract shape info
		virtual tensorshape shape_eval (void) = 0;
		
		// CONSTRUCTS THE GRADIENT TREE AND STORE ROOT IN MEMBER GRAD
		virtual void setup_gradient (void) = 0;

		void copy (const ioperation<T>& other, std::string name = "");
		
		virtual ivariable<T>* clone_impl (std::string name) = 0;

		// used specifically to pass jacobian tensors up the tree... could make generic use in the future
		virtual bool channel (std::stack<ivariable<T>*>& jacobi) {
			// propagate channel
			// did not implement jacobian conflicts resolution (when 2 jacobian nodes meeting at the same junction...)
			// as such, this is undefined behavior for now and should throw error
			size_t jacobi_count = 0;
			for (ccoms::subject* sub : this->dependencies_) {
				if (ioperation<T>* o = dynamic_cast<ioperation<T>*>(sub)) {
					if (o->channel(jacobi)) {
						jacobi_count++;
					}
				}
			}
			if (jacobi_count > 1) {
				throw std::logic_error("jacobian branch conflict occurred at " + this->get_name());
			}
			return jacobi_count != 0;
		}

		ioperation (const ioperation<T>& other, std::string name) :
			ivariable<T>(other, name),
			valid_tensor_(other.valid_tensor_),
			grad_(other.grad_) {}

		friend class gradient<T>;

	public:
		ioperation (std::vector<ivariable<T>*> dependencies, std::string name);

		virtual ~ioperation (void);

		// COPY
		ioperation<T>* clone (std::string name = "") {
			return static_cast<ioperation<T>*>(clone_impl(name));
		}
		virtual ioperation<T>& operator = (const ioperation<T>& other);
		
		virtual tensor<T>* get_eval (void) {
			if (false == valid_tensor_) {
				return nullptr;
			}
			return this->out_.get();
		}

		virtual ivariable<T>* get_gradient (void) {
			if (nullptr == grad_) {
				setup_gradient();
			}
			return grad_;
		}
};

}

#include "../../../src/graph/operation/ioperation.ipp"

#endif /* operation_hpp */
