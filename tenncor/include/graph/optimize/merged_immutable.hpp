/*!
 *
 *  merged_immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  merge connector and its children
 *
 *  Created by Mingkai Chen on 2017-05-23.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef TENNCOR_MERGED_IMMUTABLE_HPP
#define TENNCOR_MERGED_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
class merged_immutable : public immutable<T>
{
public:
	//! builder for merged_immutables
	static merged_immutable<T>* get (immutable<T>* conn);

	// >>>> CLONE, COPY && MOVE <<<<
	//! clone function
	merged_immutable<T>* clone (void) const;

	//! move function
	merged_immutable<T>* move (void);

	//! summarize this connector
	virtual void summarize (std::vector<typename iconnector<T>::conn_summary>& conn_list) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! merged_immutable constructor merging connector and its children
	merged_immutable (immutable<T>* conn);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

	//! forward pass step: populate data_ (overridden by merged_immutable)
	virtual void forward_pass (std::vector<const tensor<T>*> tens);

	//! backward pass step: populate gcache_[leaf] (overridden by merged_immutable)
	virtual void backward_pass (std::vector<inode<T>*> deps, variable<T>* leaf);

private:
	std::vector<typename iconnector<T>::conn_summary> summaries_;

	std::vector<size_t> top_deps_;
};

}

#include "../../../src/graph/optimize/merged_immutable.ipp"

#endif /* TENNCOR_MERGED_IMMUTABLE_HPP */
