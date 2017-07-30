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

#include "graph/react/iobserver.hpp"
#include "graph/inode.hpp"
#include "graph/leaf/placeholder.hpp"

#pragma once
#ifndef TENNCOR_VARPTR_HPP
#define TENNCOR_VARPTR_HPP

namespace nnet
{

template <typename T>
class varptr : public iobserver
{
public:
	void* operator new (size_t) = delete;

	//! nullptr construction
	varptr (void) {}

	//! wrap ptr construction
	varptr (inode<T>* ptr);

	virtual ~varptr (void) {}

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
	
	virtual void update (std::unordered_set<size_t>) {}

	void clear (void) { this->remove_dependency(0); }
	
protected:
	virtual void death_on_broken (void)
	{
		if (false == this->dependencies_.empty())
			this->remove_dependency(0);
	}
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
