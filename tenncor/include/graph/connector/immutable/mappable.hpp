/*!
 *
 * 	mappable.hpp
 * 	cnnet
 *
 * 	Purpose:
 * 	declares a node that allows all values along a specified dimension to act as a single element
 * 	that is a mappable A of shape <4, 4, 4> with dimensional index of 1
 * 	can elementary operate with a node B of shape <4, 1, 4> (or <4, 4> once flexible shapes are implemented [todo])
 * 	by mapping elements of coordinate
 * 		for i=0:3
 * 			output<x, i, z> = (A<x, i, z>, B<x, 1, z>)
 * 	where F is the elementary operator
 *
 * 	Created by Mingkai Chen on 2017-05-09.
 * 	Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/immutable.hpp"

#ifndef TENNCOR_MAPPABLE_HPP
#define TENNCOR_MAPPABLE_HPP

namespace nnet
{

template <typename T>
class mappable : public immutable<T>
{
public:
	//! mappable performs no forward or backward operation (it is an identity operation)
	static mappable<T>* get (inode<T>* arg, size_t idx);

	// >>>> CLONE, COPY && MOVE <<<<
	//! Clone function
	mappable<T>* clone (void) const;

	//! Move function
	mappable<T>* move (void);

protected:
	// >>>> CONSTRUCTOR <<<<
	//! identity
	mappable (inode<T>* arg, size_t idx);

	//! Implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);
};

}

#include "../../../../src/graph/connector/immutable/mappable.ipp"

#endif //TENNCOR_MAPPABLE_HPP
