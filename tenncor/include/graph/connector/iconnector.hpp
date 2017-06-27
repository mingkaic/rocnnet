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

#include "graph/varptr.hpp"
#include "graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_ICONNECTOR_HPP
#define TENNCOR_ICONNECTOR_HPP

namespace nnet
{

//! backward transfer function, get gradient nodes; F: Nf -> Nb
template <typename T>
using BACK_MAP = std::function<varptr<T>(std::vector<inode<T>*>,variable<T>*)>;

//! jacobian transfer function
template <typename T>
using JTRANSFER = std::function<inode<T>*(inode<T>*,variable<T>*)>;

template <typename T>
class iconnector : public inode<T>, public iobserver
{
public:
	//! Virtual destructor for deleting from iconnector
	virtual ~iconnector (void);

	// >>>> CLONE, COPY && MOVE ASSIGNMENTS <<<<
	//! clone function
	iconnector<T>* clone (void) const;

	//! move function
	iconnector<T>* move (void);

	//! Declare copy assignment to enforce proper gid_ copy over
	virtual iconnector<T>& operator = (const iconnector<T>& other);

	//! Declare move assignment to enforce proper gid_ copy over
	virtual iconnector<T>& operator = (iconnector<T>&& other);

	// >>>> META DATA ACCESSORS <<<<
	//! Get unique label with arguments
	virtual std::string get_name (void) const;

	virtual std::vector<inode<T>*> get_arguments (void) const
	{
		std::vector<inode<T>*> node_args(this->dependencies_.size());
		std::transform(this->dependencies_.begin(), this->dependencies_.end(), node_args.begin(),
			[](subject* s) { return static_cast<inode<T>*>(s); });
		return node_args;
	}

	virtual size_t n_arguments (void) const { return this->dependencies_.size(); }

	// >>>> GRAPH ACCESSOR <<<<
	//! Check if other connector is in the same graph as this
	bool is_same_graph (const iconnector<T>* other) const;

	//! check if connector n is a potential descendent of this node
	virtual bool potential_descendent (const iconnector<T>* n) const;

	//! Grab a temporary value traversing top-down
	virtual void temporary_eval (const iconnector<T>* target, inode<T>*& out) const = 0;

	// >>>> GRAPH WIDE OPTION <<<<
	//! Freeze or unfreeze the entire graph
	//! Freeze prevents graph from updating temporarily (updates are queued)
	//! Unfreeze allows graph to be updated again, and executes all updates in queue
	void update_status (bool freeze);

	// >>>> STRUCTS <<<<
	//! iconnector summary
	struct conn_summary
	{
		conn_summary (std::string id, transfer_func<T>* forward, BACK_MAP<T> back) :
			id_(id), Nf_(forward->clone()), ginit_(back) {}

		std::string id_;

		transfer_func<T>* Nf_;

		BACK_MAP<T> ginit_;

		std::vector<std::string> arg_ids_;
	};

	using summary_series = std::vector<typename iconnector<T>::conn_summary>;

	//! Summarize this connector
	virtual summary_series summarize (void) const = 0;

protected:
	//! list of jacobian transfer function
	//! to be executed on resulting root node
	//! execution order: top-down
	struct JList
	{
		JList (void) : uid_(nnutils::uuid(this)) {}

		std::string uid_;
		std::list<JTRANSFER<T> > list_;
	};

	//! graph info shareable between connectors
	struct graph_node;

	// >>>> CONSTRUCTORS <<<<
	//! Set dependencies
	iconnector (std::vector<inode<T>*> dependencies, std::string label);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! Declare copy constructor to enforce proper gid_ copy over
	iconnector (const iconnector<T>& other);

	//! Declare move constructor to enforce proper gid_ copy over
	iconnector (iconnector<T>&& other);

	//! Update gid_ by updating all argument variables
	virtual void update_graph (std::vector<iconnector<T>*> args);

	// >>>> SPECIALIZED OPERATOR FOR ALL CONNECTORS <<<<
	//! jacobian for each variable
	using JCACHE = std::unordered_map<variable<T>*,JList>;
	JCACHE jacobians_;

	// >>>> GRAPH META DATA <<<<
	graph_node* gid_ = nullptr; //! graph hash
};

}

#include "../../../src/graph/connector/iconnector.ipp"

#endif /* TENNCOR_ICONNECTOR_HPP */
