/*!
 *
 *  default_alloca.hpp
 *  cnnet
 *
 *  Purpose:
 *  default_allocator implements iallocator by
 *  wrapping the OS default allocator
 *
 *  Created by Mingkai Chen on 2016-11-11.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 */

#include "memory/iallocator.hpp"

#ifndef TENNCOR_DEFAULT_ALLOC_HPP
#define TENNCOR_DEFAULT_ALLOC_HPP

namespace nnet
{

class default_alloc : public iallocator
{
public:
	//! identifier for builder
	static const size_t alloc_id;

	//! clone function
	default_alloc* clone (void) const;

	//! move function
	default_alloc* move (void);

protected:
	//! allocation implementation
	virtual void* get_raw (size_t alignment, size_t num_bytes);

	//! deallocation implementation
	virtual void del_raw (void* ptr, size_t num_bytes);

	//! clone implementation
	virtual iallocator* clone_impl (void) const;

	//! move implementation
	virtual iallocator* move_impl (void);
};

}

#endif /* TENNCOR_DEFAULT_ALLOC_HPP */
