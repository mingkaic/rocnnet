//
//  igraph.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/iconnector.hpp"

#pragma once
#ifndef igraph_hpp
#define igraph_hpp

namespace nnet
{

template <typename T>
std::vector<ivariable<T>*> root_dep_extract (ivariable<T>* root)
{
	std::vector<ivariable<T>*> deps = {root};
	if (iconnector<T>* connect = dynamic_cast<iconnector<T>*>(root))
	{
		connect->leaves_collect([&deps](ivariable<T>* leaf)
		{
			deps.push_back(leaf);
		});
	}
	return deps;
}

// takes the root as its first dependency
// almost like buffer except graph takes leaves of dependencies (or some other variable)
// as additional dependencies
// TODO make igraph child of buffer, since graph treats root exactly identically to buffer's dependency
template <typename T>
class igraph : public iconnector<T>
{
	protected:
		igraph (const igraph<T>& other, std::string name) : iconnector<T>(other, name) {}

		virtual ivariable<T>* clone_impl (std::string name) = 0;

		// auto extract leaves of root as dependencies
		// THIS could be an issue if graph leaves are changing (due to buffer)
		// TODO consider better leaf extraction that accommodates changing leaves
		igraph (ivariable<T>* root) :
			iconnector<T>(root_dep_extract(root), "graph<" + root->get_name() + ">") {}

		// the safest constructor, takes leaf as a guaranteed dependency
		igraph (ivariable<T>* root, ivariable<T>* leaf) :
			iconnector<T>(std::vector<ivariable<T>*>{root, leaf}, "graph<" + root->get_name() + ">") {}

	public:
		virtual ~igraph (void) {}

		// COPY
		igraph* clone (std::string name = "")
		{
			return static_cast<igraph*>(clone_impl(name));
		}

		// unique to graph
		ivariable<T>* get_root (void) const
		{
			return sub_to_var<T>(this->dependencies_[0]);
		}

		// methods specific to leaf manipulation
		virtual void connect_graph (igraph<T>* g_other) = 0;
		virtual void update_leaf (std::function<ivariable<T>*(ivariable<T>*,size_t)> lassign) = 0;

		virtual tensorshape get_shape (void) const { return get_root()->get_shape(); }
		// special jacobian: eval the leaf instead of root
		virtual tensor<T>* get_eval (void) { return get_root()->get_eval(); }
		// jacobian special: evaluate leaf
		virtual ivariable<T>* get_gradient (void) { return get_root()->get_gradient(); }

		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			size_t callerid = info.caller_idx_;
			// ignore leaf updates
			// leaves update propagates to root, then root updates this
			if (callerid != 0) return;
			this->notify(msg);
		}
		
		virtual igraph<T>* get_jacobian (void)
		{
			if (iconnector<T>* c = dynamic_cast<iconnector<T>*>(get_root()))
			{
				return c->get_jacobian();
			}
			return nullptr;
		}
};

}

#endif /* igraph_hpp */
