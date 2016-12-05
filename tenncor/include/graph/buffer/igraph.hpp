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

		// abstracts from ivariable
		// get_shape remains abstract
		// get_eval remains abstract
		// get_gradient remains abstract
		// update remains abstract
};

}

#endif /* igraph_hpp */
