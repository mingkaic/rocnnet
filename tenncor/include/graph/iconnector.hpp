//
//  iconnector.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/iobserver.hpp"
#include "executor/varptr.hpp"

#pragma once
#ifndef iconnector_hpp
#define iconnector_hpp

namespace nnet
{

template <typename T>
class gradient;

// operates on nodes
template <typename T>
class iconnector : public ivariable<T>, public ccoms::iobserver
{
	private:
		// remember that once leaf subjects are destroyed,
		// everyone in this graph including this is destroyed
		// so we don't need to bother with cleaning leaves_
		std::unordered_set<ivariable<T>*> leaves_;

	protected:
		virtual void merge_leaves (std::unordered_set<ivariable<T>*>& src);
		
		// CONSTRUCTOR
		iconnector (std::vector<ivariable<T>*> dependencies, std::string name);

		friend class gradient<T>;

	public:
		virtual ~iconnector (void) {}
		
		// COPY
		// abstract clone

		// connectors only
		void leaves_collect (std::function<void(ivariable<T>*)> collector);

		// abstracts from ivariable
		// get_shape remains abstract
		// get_eval remains abstract
		// get_gradient remains abstract
		virtual functor<T>* get_jacobian (void) = 0;
		// update remains abstract
};

}

#include "../../src/graph/iconnector.ipp"

#endif /* iconnector_hpp */
