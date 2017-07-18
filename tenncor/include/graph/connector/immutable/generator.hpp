/*!
 *
 *  generator.hpp
 *  cnnet
 *
 *  Purpose:
 *  generate values using init given shape dependency on shape_dep node
 *
 *  Created by Mingkai Chen on 2017-07-18.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/iconnector.hpp"

#pragma once
#ifndef ROCNNET_GENERATOR_HPP
#define ROCNNET_GENERATOR_HPP

namespace nnet
{

template <typename T>
class generator : public iconnector<T>
{
public:
	virtual ~generator (void) {}

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for generator, clones init
	static generator<T>* get (inode<T>* shape_dep,
		const initializer<T>& init, std::string name = "generator")
	{
		return new generator(shape_dep, init, name);
	}

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	generator<T>* clone (void) const;

	//! move function
	generator<T>* move (void);

	//! declare copy assignment to copy over ?
	virtual generator<T>& operator = (const generator<T>& other);

	//! declare move assignment to move over ?
	virtual generator<T>& operator = (generator<T>&& other);

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! grab a temporary value traversing top-down
	//! allocates out tensor. caller owns out
	virtual void temporary_eval (const iconnector<T>* target, inode<T>*& out) const;

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr<T> derive (inode<T>* wrt);

	//! Utility function: get data shape
	virtual tensorshape get_shape (void) const;

	// >>>> GRAPH STATUS <<<<
	//! get gradient leaves
	virtual std::unordered_set<ileaf<T>*> get_leaves (void) const
	{
		return {};
	}

	// >>>> NODE STATUS <<<<
	//! check if the arguments are good; data is available
	virtual bool good_status (void) const;

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&) {}

	// >>>> CALLED BY OBSERVER TO UPDATE <<<<
	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (std::unordered_set<size_t> argidx) {}

protected:
	generator (inode<T>* shape_dep, const initializer<T>& init, std::string name) :
		iconnector<T>({shape_dep}, name) {}

	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone abstraction function
	virtual inode<T>* clone_impl (void) const;

	//! move abstraction function
	virtual inode<T>* move_impl (void);

	virtual const tensor<T>* get_eval (void) const;

	virtual inode<T>* get_gradient (variable<T>* leaf);

	virtual void death_on_broken (void);

	virtual typename iconnector<T>::summary_series summarize (void) const;

private:
};

}

#endif /* ROCNNET_GENERATOR_HPP */
