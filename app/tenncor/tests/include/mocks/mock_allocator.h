//
// Created by Mingkai Chen on 2017-03-12.
//

#ifndef TENNCOR_MOCK_ALLOCATOR_H
#define TENNCOR_MOCK_ALLOCATOR_H

#include <algorithm>

#include "util_test.h"

#include "memory/iallocator.hpp"
#include "memory/default_alloc.hpp"

using namespace nnet;


class mock_default_allocator : public default_alloc
{
public:
	struct memoryinfo
	{
		size_t num_bytes;
		size_t alignment;
	};

	std::unordered_map<void*,memoryinfo> tracker;

protected:
	virtual void* get_raw (size_t alignment, size_t num_bytes)
	{
		void* ptr = default_alloc::get_raw(alignment, num_bytes);
		tracker[ptr] = {num_bytes, alignment};
		return ptr;
	}

	virtual void del_raw (void* ptr, size_t num_bytes)
	{
		tracker.erase(ptr);
		default_alloc::del_raw(ptr, num_bytes);
	}
};


class mock_allocator : public iallocator
{
public:
	mock_allocator (void) :
		tracksize_(false) {}

	mock_allocator (bool tracksize) :
		tracksize_(tracksize) {}

	virtual bool tracks_size (void) const
	{
		return tracksize_;
	}

	virtual size_t requested_size (void* ptr) const
	{
		if (false == tracksize_)
		{
			iallocator::requested_size(ptr);
		}
		char* cptr = (char*)ptr;
		auto it = tracker.find(cptr);
		if (it == tracker.end())
		{
			return 0;
		}
		return it->second.num_bytes;
	}

	virtual optional<size_t> alloc_id (void* ptr) const
	{
		optional<size_t> id;
		if (false == tracksize_)
		{
			id = iallocator::alloc_id(ptr);
		}
		return id;
	}

	size_t uid = 0;
	bool tracksize_ = false;
	struct memoryinfo
	{
		bool master;
		size_t num_bytes;
		size_t alignment;
	};

	std::unordered_map<char*, memoryinfo> tracker;
	default_alloc realalloc;

protected:
	virtual void* get_raw (size_t alignment, size_t num_bytes)
	{
		char* ptr = new char('A');
		tracker[ptr] = {true, num_bytes, alignment};
		if (tracksize_)
		{
			for (size_t i = 1; i < num_bytes; i++)
			{
				tracker[ptr+i] = {false, num_bytes - i, alignment};
			}
		}
		return ptr;
	}

	virtual void del_raw (void* ptr, size_t num_bytes)
	{
		char* cptr = (char*) ptr;
		if (false == tracksize_)
		{
			// remove from tracker
			tracker.erase(cptr);
			delete cptr;
		}
		else
		{
			auto it = tracker.find(cptr);
			size_t end = it->second.num_bytes;
			if (end < num_bytes)
			{
				throw std::exception();
			}
			for (size_t i = 1; i < num_bytes; i++)
			{
				auto t = tracker.find(cptr+i);
				if (t == tracker.end())
				{
					throw std::exception();
				}
				tracker.erase(t);
			}
			if (false == it->second.master)
			{
				tracker.erase(it);
				// back trace to master
				if (end == num_bytes)
				{
					size_t i = 0;
					bool lone = true;
					bool foundmaster = false;
					while (false == foundmaster)
					{
						i++;
						auto t = tracker.find(cptr - i);
						if (tracker.end() != t)
						{
							foundmaster = t->second.master;
							if (!foundmaster)
							{
								lone = false;
							}
						}
					}
					if (lone)
					{
						tracker.erase(cptr - i);
						delete (cptr - i);
					}
				}
			}
			else
			{
				// erase
			}
		}
	}

	virtual iallocator* clone_impl (void) const
	{
		return new mock_allocator(*this);
	}

	virtual iallocator* move_impl (void)
	{
		return new mock_allocator(*this);
	}
};


#endif //TENNCOR_MOCK_ALLOCATOR_HPP
