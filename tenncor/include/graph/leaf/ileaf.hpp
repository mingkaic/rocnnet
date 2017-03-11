/*!
 *
 *  ileaf.hpp
 *  cnnet
 *
 *  Purpose:
 *  leaf interface abstractly defines
 *  all pure subject nodes in the graph
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/inode.hpp"

#pragma once
#ifndef TENNCOR_ILEAF_HPP
#define TENNCOR_ILEAF_HPP

namespace nnet
{

template <typename T>
class ileaf : public inode<T>
{
public:
	//! kill handler
	virtual ~ileaf (void);

	// >>>> CLONE, COPY && MOVE ASSIGNMENTS <<<<
	//! clone function
	ileaf<T>* clone (void) const;

	//! declare copy assignment to copy over initializer
	virtual ileaf<T>& operator = (const ileaf<T>& other);

	//! declare move assignment to move over initializer
	virtual ileaf<T>& operator = (ileaf<T>&& other);

	// >>>> ACCESSORS <<<<
	//! utility function: get data shape
	virtual tensorshape get_shape (void) const;

	//! get forward passing value
	//! return nullptr if leaf is not init
	virtual const tensor<T>* get_eval (void) const;

	//! determine whether leaf node can be initiated
	bool can_init (void) const;

	//! check if data is available
	//! (if the node is initialized)
	virtual bool good_status (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! assign initializer
	ileaf (const tensorshape& shape,
		itensor_handler<T>* init,
		std::string name);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! declare copy constructor to copy over initializer
	ileaf (const ileaf<T>& other);

	//! declare move constructor to move over initializer
	ileaf (ileaf<T>&& other);

	// >>>> TENSOR CONTENT <<<<
	//! tensor data
	std::unique_ptr<tensor<T> > data_ = nullptr;

	// >>>> INITIALIZER DATA <<<<
	//! tensor manipulator, ileaf owns this
	itensor_handler<T>* init_ = nullptr;

	//! tensor state (good or bad) true = good
	bool is_init_ = false;

	//! dynamically initialize tensors
	//! used by placeholder
	struct assignment;

private:
	//! move helper
	void move (ileaf<T>&& other);

	//! copy helper
	void copy (const ileaf<T>& other);
};

}

#include "../../../src/graph/leaf/ileaf.ipp"

#endif /* TENNCOR_ILEAF_HPP */
