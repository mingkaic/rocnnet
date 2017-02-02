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

		// add argument names to name
		virtual std::string get_name (void) const
		{
			std::string args;
			access_dependencies(
			[&args](const ccoms::subject_owner* subs)
			{
				if (const ivariable<T>* arg = dynamic_cast<const ivariable<T>*>(subs))
				{
					args += arg->get_label() + ",";
				}
			});
			if (!args.empty()) args.pop_back();
			return ivariable<T>::get_name() + "(" + args + ")";
		}

		// abstracts from ivariable
		// get_shape remains abstract
		// get_eval remains abstract
		// get_gradient remains abstract
		virtual functor<T>* get_jacobian (void) = 0;
		// update remains abstract

		virtual void get_args (std::vector<ivariable<T>*>& args) const
		{
			args.clear();
			for (ccoms::subject* sub : this->dependencies_)
			{
				args.push_back(sub_to_var<T>(sub));
			}
		}
};

}

#include "../../src/graph/iconnector.ipp"

#endif /* iconnector_hpp */
