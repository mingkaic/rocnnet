//
//  assign.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "iexecutor.hpp"
#define assign_hpp
#pragma once
#ifndef assign_hpp
#define assign_hpp

namespace nnet
{

template <typename T>
using ASSIGN_OP = std::function<void(T&,T)>;

template <typename T>
void direct (T& dest, T src) { dest = src; }

// NON-REACTIVE OPERATION

template <typename T>
class assign : public iexecutor<T>
{
	private:
		// determines how element-wise assignment works, defaults to direct assignment
		ASSIGN_OP<T> transfer_;
		// target (weak pointer, no ownership)
		variable<T>* dest_;
		std::vector<T> local_cpy_;

	protected:
		// used by copy constructor and copy assignment
		void copy (const assign<T>& other);
		assign (const assign<T>& other);
		virtual iexecutor<T>* clone_impl (void);

	public:
		assign (variable<T>* dest, 
			inode<T>* src,
			ASSIGN_OP<T> trans = direct);
		
		// COPY
		assign<T>* clone (void);
		assign<T>& operator = (const assign<T>& other);
		
		// MOVE

		// inherited from iexecutor
		virtual void freeze (void);
		virtual void execute (void);
};

template <typename T>
class assign_sub : public assign<T>
{
	protected:
		assign_sub (const assign_sub<T>& other);
		virtual assign<T>* clone_impl (void);

	public:
		assign_sub (variable<T>* dest, inode<T>* src);

		// COPY
		assign_sub<T>* clone (void);
		
		// MOVE
};

}

#include "../../src/executor/assign.ipp"

#endif /* assign_hpp */
