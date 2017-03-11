//
//  iallocator.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "memory/iallocator.hpp"

#ifdef TENNCOR_ALLOCATOR_HPP

namespace nnet
{
	
iallocator* iallocator::clone (void) const {
	return clone_impl();
}

template <typename T>
T* iallocator::allocate (size_t num_elements) const
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

template <typename T>
void iallocator::dealloc(T* ptr, size_t num_elements) const
{
	if (nullptr != ptr)
	{
		del_raw(ptr, sizeof(T) * num_elements);
	}
}


}

#endif
