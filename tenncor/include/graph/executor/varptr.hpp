//
//  varptr.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-13.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/variable/placeholder.hpp"

#pragma once
#ifndef varptr_hpp
#define varptr_hpp

namespace nnet
{

// tensor variable pointer wrapper

template <typename T>
class varptr
{
	private:
		ivariable<T>* ptr_;

	public:
		varptr (ivariable<T>* ptr);
		varptr<T>& operator = (ivariable<T>* other);
		varptr<T>& operator = (const varptr<T>& other);

		explicit operator ivariable<T>* (void) const;
		ivariable<T>& operator * (void);
		ivariable<T>* operator -> (void);
		
		ivariable<T>* get (void) const;
};

template <typename T>
class placeptr
{
	private:
		placeholder<T>* ptr_;

	public:
		placeptr (placeholder<T>* ptr);
		placeptr<T>& operator = (placeholder<T>* other);
		placeptr<T>& operator = (const placeptr<T>& other);

		placeptr<T>& operator = (std::vector<T> vec);
		placeptr<T>& operator = (const tensor<T>& ten);

		explicit operator placeholder<T>* (void) const;
		placeholder<T>& operator * (void);
		placeholder<T>* operator -> (void);
		
		placeholder<T>* get (void) const;
};

}

#include "../../../src/graph/executor/varptr.ipp"

#endif /* varptr_hpp */
