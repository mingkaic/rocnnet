/*!
 *
 *  varptr.hpp
 *  cnnet
 *
 *  Purpose:
 *  varptr wraps variable pointers
 *  in order to overload operators for connectors
 *
 *  Created by Mingkai Chen on 2016-11-13.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/leaf/placeholder.hpp"

#pragma once
#ifndef TENNCOR_VARPTR_HPP
#define TENNCOR_VARPTR_HPP

namespace nnet
{

// TODO: override delete to destroy ptr directly
template <typename T>
class varptr
{
public:
	//! nullptr construction
	varptr (void) {}

	//! wrap ptr construction
	varptr (inode<T>* ptr);

	//! assign another wrapper
	varptr<T>& operator = (const varptr<T>& other);

	//! assign ptr
	varptr<T>& operator = (inode<T>* other);

	//! implicitly converts to inode<T>*
	operator inode<T>* () const;

	//! dereference overload
	inode<T>& operator * (void) const;

	//! pointer member access overload
	inode<T>* operator -> (void) const;

	//! get inner pointer
	inode<T>* get (void) const;

protected:
	//! inner pointer, not owned by this
	inode<T>* ptr_ = nullptr;
};

template <typename T>
class placeptr : public varptr<T>
{
	public:
		//! nullptr construction
		placeptr (void) {}

		//! wrap placeholder pointer
		placeptr (placeholder<T>* ptr);

		//! assign a pointer
		placeptr<T>& operator = (placeholder<T>* other);

		//! assign a wrapper
		placeptr<T>& operator = (const placeptr<T>& other);

		// >>>> EXTENDING PLACEHOLDER <<<<
		//! assign a raw data
		placeptr<T>& operator = (std::vector<T> vec);

		//! assign a tensor
		placeptr<T>& operator = (tensor<T>& ten);

		//! implicit pointer conversion
		operator placeholder<T>* () const;

		//! dereference overload
		placeholder<T>& operator * (void);

		//! pointer accessor overload
		placeholder<T>* operator -> (void);

		//! get inner pointer as placeholder pointer
		placeholder<T>* get (void) const;
};

}

#include "../../src/graph/varptr.ipp"

#endif /* TENNCOR_VARPTR_HPP */
