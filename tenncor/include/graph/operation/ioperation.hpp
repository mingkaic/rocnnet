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

#include "graph/variable/variable.hpp"
#include "graph/ccoms/iobserver.hpp"
#include "executor/varptr.hpp"

#pragma once
#ifndef operation_hpp
#define operation_hpp

namespace nnet
{

template <typename T>
using BUILD_DERIVE = std::function<ivariable<T>*(std::vector<ivariable<T>*>)>;

template <typename T>
class gradient;

template <typename T>
class jacobian;

// INTERFACE OPERATION

template <typename T>
class ioperation : public ivariable<T>, public ccoms::iobserver
{
	private:
		// remember that once leaf subjects are destroyed,
		// everyone in this graph including this is destroyed
		// so we don't need to bother with cleaning leaves_
		std::unordered_set<ivariable<T>*> leaves_;

	protected:
		bool valid_tensor_ = false;
		ioperation<T>* grad_ = nullptr;

		virtual void merge_leaves (std::unordered_set<ivariable<T>*>& src)
		{
			src.insert(this->leaves_.begin(), this->leaves_.end());
		}

		// implement unique method of consuming input variables
		// to extract shape info
		virtual tensorshape shape_eval (void) = 0;
		
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

	public:
		virtual ~ioperation (void);

		// COPY
		ioperation<T>* clone (std::string name = "");
		virtual ioperation<T>& operator = (const ioperation<T>& other);
		
		// MOVE
		
		// inherited from ioperation
		virtual tensor<T>* get_eval (void); // override
		
		// non-inherited
		virtual ivariable<T>* get_gradient (void);

		// operations only
		void leaves_collect (std::function<void(ivariable<T>*)> collector);
};

}

#include "../../../src/graph/operation/ioperation.ipp"

#endif /* operation_hpp */
