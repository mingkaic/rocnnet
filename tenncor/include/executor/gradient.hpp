//
//  gradient.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "iexecutor.hpp"
#include "graph/variable/constant.hpp"
#include "tensor/tensor_jacobi.hpp"

#pragma once
#ifndef gradient_hpp
#define gradient_hpp

namespace nnet
{
	
template <typename T>
using GRAD_GATHER = std::function<void(ivariable<T>*,placeholder<T>*)>;
template <typename T>
using GRAD_MAP = std::unordered_map<ivariable<T>*, placeholder<T>*>;

template <typename T>
class gradient : public iexecutor<T>
{
	private:
		// gradient owns values in leaf_map_
		ivariable<T>* g_root_;
		std::vector<ccoms::subject*> potential_srcs_;
		GRAD_MAP<T> leaf_map_;

	protected:
		void clear_map (void);

		void copy (const gradient<T>& other);
		gradient (const gradient<T>& other);
		virtual iexecutor<T>* clone_impl (void);

	public:
		gradient (ivariable<T>* root, ivariable<T>* leaf = nullptr);
		virtual ~gradient (void);

		// COPY
		gradient<T>* clone (void);
		gradient<T>& operator = (const gradient<T>& other);

		// MOVE

		// inherited from iexecutor
		virtual void freeze (void);
		virtual void execute (void);

		varptr<T> get_root (void) { return g_root_; }
		void collect_grad (GRAD_GATHER<T> collector);
};

}

#include "../../src/executor/gradient.ipp"

#endif /* gradient_hpp */
