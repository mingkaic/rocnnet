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
	//! virtual destructor for variable and placeholder
	virtual ~ivariable (void) {}

	// >>>> CLONE, COPY, & MOVE ASSIGNMENTS <<<<
	//! clone function
	virtual ivariable<T>* clone (void) const;

	// >>>> PUBLICLY ACCESSIBLE GRADIENT <<<<
	//! get gradient wrt some node
	virtual const tensor<T>* get_gradient (inode<T>* wrt) const;

protected:
	const constant<T> zero; //! commonly used constant zero

	const constant<T> one; //! commonly used constant one

	// >>>> CONSTRUCTORS <<<<
	//! construct to init zero and one
	ivariable (const tensorshape& shape,
		itensor_handler<T>* init,
		std::string name);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! copy construct to init zero and one
	ivariable (const ileaf<T>& other);

	//! move construct to init zero and one
	ivariable (ileaf<T>&& other);
};

}

#include "../../../src/graph/leaf/ivariable.ipp"

#endif /* TENNCOR_IVARIABLE_HPP */
