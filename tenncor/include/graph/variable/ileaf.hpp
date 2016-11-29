//
//  ileaf.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ivariable.hpp"

#pragma once
#ifndef ileaf_hpp
#define ileaf_hpp

namespace nnet
{

// INITIALIZER MANAGING INTERFACE
// Leaf Nodes

template <typename T>
class ileaf : public ivariable<T>
{
	protected:
		// used by assignment operators to dynamically initialize tensors
		struct dyn_init;

		// TODO make suicide an option for constants
		std::unique_ptr<variable<T> > grad_ = nullptr; // make it variable to prevent self destruction when disconnecting
		
		// we own our initializer
		initializer<T>* init_ = nullptr;
		bool is_init_ = false;

		void copy (const ileaf<T>& other, std::string name = "");
		ileaf (const ileaf<T>& other, std::string name);
		virtual ivariable<T>* clone_impl (std::string name) = 0;

		ileaf (const tensorshape& shape, initializer<T>* init, std::string name);

	public:
		virtual ~ileaf (void);
		
		// COPY
		// call abstract cloner
		ileaf<T>* clone (std::string name = "");
		virtual ileaf<T>& operator = (const ileaf<T>& other);

		// MOVES
		// todo: implement move clone

		// inherited from ivariable
		virtual tensor<T>* get_eval (void)
		{
			if (false == is_init())
			{
				return nullptr;
			}
			return ivariable<T>::get_eval();
		}

		// GET INFO
		bool can_init (void) const;
		bool is_init (void) const { return is_init_; }
};

}

#include "../../../src/graph/variable/ileaf.ipp"

#endif /* ileaf_hpp */
