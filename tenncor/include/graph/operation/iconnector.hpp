/*!
 *
 *  iconnector.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph connector interface
 *  manages graph information
 *
 *  Created by Mingkai Chen on 2016-12-01.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/react/iobserver.hpp"
#include "graph/varptr.hpp"
#include "graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_ICONNECTOR_HPP
#define TENNCOR_ICONNECTOR_HPP

namespace nnet
{

template <typename T>
class iconnector : public inode<T>, public iobserver
{
public:
	//! Virtual destructor for deleting from iconnector
	virtual ~iconnector (void) {}

	// >>>> CLONE, COPY && MOVE ASSIGNMENTS <<<<
	//! clone function
	iconnector<T>* clone (void) const;

	//! Declare copy assignment to prevent gid_ copy over
	virtual iconnector<T>& operator = (const iconnector<T>& other);

	//! Declare move assignment to prevent gid_ copy over
	virtual iconnector<T>& operator = (iconnector<T>&& other);

	// >>>> META DATA ACCESSORS <<<<
	//! Get unique label with arguments
	virtual std::string get_name (void) const;

	// >>>> GRAPH ACCESSOR <<<<
	//! Check if other connector is in the same graph as this
	bool is_same_graph (const iconnector<T>* other) const;

	//! check if connector n is a potential descendent of this node
	virtual bool potential_descendent (iconnector<T>* n) const;

	//! Grab a temporary value traversing top-down
	virtual void temporary_eval (const iconnector<T>* target,
		tensor<T>*& out) const = 0;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! Set dependencies
	iconnector (std::vector<inode<T>*> dependencies, std::string name);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! Declare copy constructor to prevent gid_ copy over
	iconnector (const iconnector<T>& other);

	//! Declare move constructor to prevent gid_ copy over
	iconnector (iconnector<T>&& other);

	//! Update gid_ by updating all argument variables
	virtual void update_graph (std::vector<inode<T>*> args);

	// >>>> GRAPH META DATA <<<<
	struct graph_node; //! graph info shareable between connectors

	graph_node* gid_ = nullptr; //! graph hash
};

}

#include "../../../src/graph/operation/iconnector.ipp"

#endif /* TENNCOR_ICONNECTOR_HPP */
