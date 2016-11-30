//
//  session.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <typeinfo>

#pragma once
#ifndef safe_ptr_hpp
#define safe_ptr_hpp

namespace nnet
{

struct safe_ptr
{
	void* ptr_;
	const std::type_info& info_;

	template<typename T>
	T *cast()
	{
		if (typeid(T) != info_)
		{
			throw std::bad_cast();
		}
		return static_cast<T *>(ptr_);
	}
};

}

#endif /* safe_ptr_hpp */
