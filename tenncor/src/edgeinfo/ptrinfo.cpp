//
// Created by Mingkai Chen on 2017-01-30.
//

#include "edgeinfo/ptrinfo.hpp"
#include <iostream>

#ifdef ptrinfo_hpp

namespace rocnnet_record
{

PUUID ptr_record::add (void* ptr)
{
	PUUID id;
	auto pit = ptrs_.find(ptr);
	if (ptrs_.end() == pit)
	{
		id = get_id();
		ptrs_.emplace(ptr, ptrinfo(id));
	}
	else
	{
		id = pit->second.id_;
		pit->second.usage_++;
	}
	return id;
}

std::experimental::optional<PUUID> ptr_record::get_hash (void* ptr)
{
	auto pit = ptrs_.find(ptr);
	std::experimental::optional<PUUID> id;
	if (ptrs_.end() != pit)
	{
		id = pit->second.id_;
	}
	else {
		for (auto ppairs : ptrs_) {
			std::cout << ppairs.first << " ";
		}
		std::cout << " >>" << ptr << "<<\n";
		std::cout << "hash not found\n";
	}
	return id;
}

bool ptr_record::remove (void* ptr)
{
	auto pit = ptrs_.find(ptr);
	if (ptrs_.end() == pit)
	{
		return false;
	}
	ptrinfo pifo = pit->second;
	if (0 == --pifo.usage_)
	{
		ptrs_.erase(pit);
		unusaged_id_.push(pifo.id_);
	}
	return true;
}

PUUID ptr_record::get_max_id (void) { return lastid; }

PUUID ptr_record::get_id (void)
{
	if (false == unusaged_id_.empty())
	{
		PUUID id = unusaged_id_.top();
		unusaged_id_.pop();
		return id;
	}
	return lastid++;
}

}

#endif