//
// Created by Mingkai Chen on 2017-01-30.
//

#include <experimental/optional>
#include <unordered_map>
#include <stack>

#pragma once
#ifndef ptrinfo_hpp
#define ptrinfo_hpp

namespace rocnnet_record
{

using PUUID = size_t;

// hashes pointers
class ptr_record
{
	public:
		// returns id
		PUUID add (void* ptr);

		std::experimental::optional<PUUID> get_hash (void* ptr);

		bool remove (void* ptr);

		PUUID get_max_id (void);

	private:
		PUUID get_id (void);

		struct ptrinfo
		{
			ptrinfo (PUUID id) : id_(id) {}

			PUUID id_;
			size_t usage_ = 1;
		};

		PUUID lastid = 0;
		std::unordered_map<void*, ptrinfo> ptrs_;
		std::stack<PUUID> unusaged_id_;
};

}

#endif /* ptrinfo_hpp */
