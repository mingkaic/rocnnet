/*!
 *
 *  alloc_builder.hpp
 *  cnnet
 *
 *  Purpose:
 *  alloc_builder is a singleton factory
 *  for registering and retrieving allocators
 *
 *  Created by Mingkai Chen on 2017-03-06.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 */

#include "memory/iallocator.hpp"
#include "memory/default_alloc.hpp"

#ifndef TENNCOR_ALLOC_BUILDER_HPP
#define TENNCOR_ALLOC_BUILDER_HPP

#include <unordered_map>

namespace nnet
{

class alloc_builder final
{
public:
	//! get singleton instance
	static alloc_builder& get_instance (void);

	// delete all copier and movers
	alloc_builder (alloc_builder const&) = delete;
	alloc_builder (alloc_builder&&) = delete;
	alloc_builder& operator = (const alloc_builder&) = delete;
	alloc_builder& operator = (alloc_builder&&) = delete;

	// >>>> ACCESSORS <<<<
	//! get allocator
	iallocator* get (size_t identifier) const;

	//! check if registry has type A at identifier
	//! this could have constructors attempt to register its class
	//! then check to validate key, value association
	template <typename A>
	bool check_registry (size_t identifier) const
	{
		auto it = registry_.find(identifier);
		if (registry_.end() == it) return false;
		return nullptr != dynamic_cast<A*>(it->second);
	}

	// >>>> MUTATORS <<<<
	//! register allocator. should ideally register in a constexpr function
	template <typename A>
	bool registertype (size_t identifier)
	{
		static_assert(std::is_base_of<iallocator, A>(),
			"alloc_builder only accepts types inheriting from iallocator");
		bool success = registry_.end() == registry_.find(identifier);
		if (success)
		{
			registry_[identifier] = new A();
		}
		return success;
	}

private:
	//! constructor
	alloc_builder (void);

	//! destructor, should die at library destruction
	~alloc_builder (void);

	//! map identifiers to registered allocator instance
	std::unordered_map<size_t,iallocator*> registry_;
};

}

#endif /* TENNCOR_ALLOC_BUILDER_HPP */
