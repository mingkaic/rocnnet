/*!
 *
 *  iallocator.hpp
 *  cnnet
 *
 *  Purpose:
 *  iallocator is an interface for defining
 *  custom allocators per tensor.
 *
 *  this allows tensors in the same graph to have
 *  different levels of security, and memory.
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 */

#pragma once
#ifndef TENNCOR_ALLOCATOR_HPP
#define TENNCOR_ALLOCATOR_HPP

#include <functional>
#include <algorithm>
#include <complex>
#include <limits>
#include <experimental/optional>

using namespace std::experimental;

namespace nnet
{

// todo: define some statistic container
struct alloc_stat
{
	void clear (void) {}
};

class iallocator
{
public:
	static constexpr size_t alloc_alignment = 32;

	//! virtual destructor, standard on all interfaces
	virtual ~iallocator (void) {}

	//! clone function
	iallocator* clone (void) const;

	//! move function
	iallocator* move (void);

	// >>>> ACCESSORS <<<<
	//! allocate the specified number of elements
	template <typename T>
	T* allocate (size_t num_elements)
	{
		static_assert(is_allowed<T>::value, "T is not an allowed type.");

		if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T)))
		{
			return nullptr;
		}

		void* p = get_raw(alloc_alignment, sizeof(T) * num_elements);
		T* typedptr = reinterpret_cast<T*>(p);
		return typedptr;
	}

	//! deallocate some number of elements from pointer
	//! (vary depending on allocator)
	template <typename T>
	void dealloc(T* ptr, size_t num_elements)
	{
		if (nullptr != ptr)
		{
			del_raw(ptr, sizeof(T) * num_elements);
		}
	}

	//! whether the implementation of allocator track the allocated size
	virtual bool tracks_size (void) const;

	//! get allocated size if tracking is enabled
	//! not tracked by default
	virtual size_t requested_size (void* ptr) const;

	//! get allocation id if tracking enabled
	virtual optional<size_t> alloc_id (void* ptr) const;

	 //! Fills in stat gatherer with statistics collected by this allocator.
	 virtual void gather_stat (alloc_stat& stats) const;

protected:
	//! allocation implementation
	virtual void* get_raw (size_t alignment, size_t num_bytes) = 0;

	//! deallocation implementation
	virtual void del_raw (void* ptr, size_t num_bytes) = 0;

	//! clone implementation
	virtual iallocator* clone_impl (void) const = 0;

	//! move implementation
	virtual iallocator* move_impl (void) = 0;

private:
	// todo: limit T to numerics by variant in c++17
	//! check which types are allowed
	// allow floats and doubles: is_trivial
	// allow complex
	// add other allowed types here
	template <typename T>
	struct is_allowed
	{
		static constexpr bool value =
			std::is_trivial<T>::value ||
			std::is_same<T, std::complex<size_t> >::value ||
			std::is_same<T, std::complex<double> >::value;
	};
};

}

#endif /* TENNCOR_ALLOCATOR_HPP */
