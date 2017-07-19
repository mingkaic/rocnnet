/*!
 *
 *  ivariable.hpp
 *  cnnet
 *
 *  Purpose:
 *  variable interface
 *
 *  Created by Mingkai Chen on 2017-02-27.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_IVARIABLE_HPP
#define TENNCOR_IVARIABLE_HPP

namespace nnet
{

template <typename T>
class ivariable : public ileaf<T>
{
public:
	virtual ~ivariable (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	ivariable<T>* clone (void) const;

	//! move function
	ivariable<T>* move (void);

	//! declare copy assignment to copy over initializer
	virtual ivariable<T>& operator = (const ivariable<T>& other);

	//! declare move assignment to move over initializer
	virtual ivariable<T>& operator = (ivariable<T>&& other);

	// >>>> BACKWARD DATA <<<<
	//! get gradient wrt some node
	virtual varptr<T> derive (inode<T>* wrt);

	// >>>> IVARIABLE SPECIAL <<<<
	//! determine whether leaf node can be initiated
	bool can_init (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! construct to init zero and one
	ivariable (const tensorshape& shape,
		initializer<T>* init,
		std::string name);

	//! copy construct to init zero and one
	ivariable (const ivariable<T>& other);

	//! move construct to init zero and one
	ivariable (ivariable<T>&& other);

	//! initialization handler, owns this
	initializer<T>* init_ = nullptr;

private:
	//! copy helper
	void copy_helper (const ivariable<T>& other);

	//! move helper
	void move_helper (ivariable<T>&& other);
};

}

#include "../../../src/graph/leaf/ivariable.ipp"

#endif /* TENNCOR_IVARIABLE_HPP */
