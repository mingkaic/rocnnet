//
//  gradient.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef gradient_hpp
#define gradient_hpp

#include "iexecutor.hpp"
#include "graph/variable/constant.hpp"

namespace nnet
{
	
template <typename T>
using GRAD_GATHER = std::function<void(ivariable<T>*,tensor<T>*)>;

template <typename T>
class gradient : public iexecutor<T>
{
	private:
		// gradient owns nothing
		ivariable<T>* g_root_;
		std::vector<ccoms::subject*> potential_srcs_;
		std::unordered_map<ivariable<T>*, tensor<T>*> leaf_map_;
		const constant<T> one_;

	protected:
		void copy (const gradient<T>& other);
		gradient (const gradient<T>& other);
		virtual iexecutor<T>* clone_impl (void);

	public:
		gradient (ivariable<T>* root, ivariable<T>* leaf = nullptr);

		// COPY
		gradient<T>* clone (void);
		gradient<T>& operator = (const gradient<T>& other);

		// MOVE
		
		// override inherited from iexecutor
		virtual void add (ivariable<T>* node);
		// inherited from iexecutor
		virtual void freeze (void);
		virtual void execute (std::function<bool(ivariable<T>*,tensor<T>*)> cb);
};

}

#include "../../../src/graph/bridge/gradient.ipp"

#endif /* gradient_hpp */
