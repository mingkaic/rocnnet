//
//  iconnector.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/variable/variable.hpp"
#include "graph/ccoms/iobserver.hpp"

#pragma once
#ifndef iconnector_hpp
#define iconnector_hpp

namespace nnet
{

template <typename T>
class gradient;

template <typename T>
class igraph;

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

		virtual ivariable<T>* clone_impl (std::string name) = 0;
		void copy (const iconnector<T>& other, std::string name = "");
		iconnector (const iconnector<T>& other, std::string name);
		iconnector (std::vector<ivariable<T>*> dependencies, std::string name);

		void leaves_update (void)
		{
			leaves_.clear();
			for (ccoms::subject* dep : this->dependencies_)
			{
				ivariable<T>* leaf = sub_to_var<T>(dep);
				leaf->merge_leaves(leaves_);
			}
		}

		friend class gradient<T>;

	public:
		virtual ~iconnector (void) {}

		iconnector<T>* clone (std::string name = "");

		// connectors only
		void leaves_collect (std::function<void(ivariable<T>*)> collector);

		// abstracts from ivariable
		// get_shape remains abstract
		// get_eval remains abstract
		// get_gradient remains abstract
		// update remains abstract
		
		virtual igraph<T>* get_jacobian (void) = 0;
};

}

#include "../../src/graph/iconnector.ipp"

#endif /* iconnector_hpp */
