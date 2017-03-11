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

#include <functional>
#include <algorithm>
#include <complex>
#include <experimental/optional>

using namespace std::experimental;

#pragma once
#ifndef TENNCOR_ALLOCATOR_HPP
#define TENNCOR_ALLOCATOR_HPP

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

	// >>>> ACCESSORS <<<<
	//! allocate the specified number of elements
	template <typename T>
	T* allocate (size_t num_elements) const;

	//! deallocate some number of elements from pointer
	//! (vary depending on allocator)
	template <typename T>
	void dealloc(T* ptr, size_t num_elements) const;

	//! whether the implementation of allocator track the allocated size
	virtual bool tracks_size (void) const { return false; }

	//! get allocated size if tracking is enabled
	//! not tracked by default
	virtual size_t requested_size (void* ptr) const
	{
		throw std::bad_function_call();
		return 0;
	}

	//! get allocation id if tracking enabled
	virtual optional<size_t> alloc_id (void* ptr) const { return optional<size_t>(); }

	 //! Fills in stat gatherer with statistics collected by this allocator.
	 virtual void gather_stat (alloc_stat& stats) const { stats.clear(); }

protected:
	//! allocation implementation
	virtual void* get_raw (size_t alignment, size_t num_bytes) const = 0;

	//! deallocation implementation
	virtual void del_raw (void* ptr, size_t num_bytes) const = 0;

	//! clone implementation
	virtual iallocator* clone_impl (void) const = 0;

private:
	// todo: remove T for variant once c++17 is supported
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
