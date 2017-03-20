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
	//! kill initializer
	virtual ~ivariable (void);

	// >>>> CLONE, COPY, & MOVE ASSIGNMENTS <<<<
	//! clone function
	ivariable<T>* clone (void) const;

	//! move function
	ivariable<T>* move (void);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! copy construct to init zero and one
	ivariable (const ivariable<T>& other);

	//! move construct to init zero and one
	ivariable (ileaf<T>&& other);

	//! declare copy assignment to copy over initializer
	virtual ivariable<T>& operator = (const ivariable<T>& other);

	//! declare move assignment to move over initializer
	virtual ivariable<T>& operator = (ivariable<T>&& other);

	//! determine whether leaf node can be initiated
	bool can_init (void) const;

	// >>>> PUBLICLY ACCESSIBLE GRADIENT <<<<
	//! get gradient wrt some node
	virtual const tensor<T>* get_gradient (inode<T>* wrt);

protected:
	std::unique_ptr<constant<T> > zero; //! commonly used constant zero
	std::unique_ptr<constant<T> > one; //! commonly used constant one

	// >>>> CONSTRUCTORS <<<<
	//! construct to init zero and one
	ivariable (const tensorshape& shape,
		initializer<T>* init,
		std::string name);

	// >>>> INITIALIZER DATA <<<<
	//! tensor manipulator, ileaf owns this
	initializer<T>* init_ = nullptr;

private:
	//! copy helper
	void copy_helper (const ivariable<T>& other);

	//! move helper
	void move_helper (ivariable<T>&& other);

	//! common init
	void common (void);
};

}

#include "../../../src/graph/leaf/ivariable.ipp"

#endif /* TENNCOR_IVARIABLE_HPP */
