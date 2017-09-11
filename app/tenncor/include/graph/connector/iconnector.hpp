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
using BACK_MAP = std::function<varptr<T>(std::vector<std::pair<inode<T>*,inode<T>*> >)>;

template <typename T>
using NODE_MAN = std::function<inode<T>*(inode<T>*)>;

//! jacobian transfer function
template <typename T>
using JTRANSFER = std::function<inode<T>*(inode<T>*,std::vector<inode<T>*>,std::vector<inode<T>*>)>;

//! calculate output shape from argument shapes
using SHAPER = std::function<tensorshape(std::vector<tensorshape>)>;

template <typename T>
class iconnector : public inode<T>, public iobserver
{
public:
	//! iconnector summary
	struct conn_summary
	{
		conn_summary (std::string id, SHAPER shaper,
			TRANSFER_FUNC<T> forward, BACK_MAP<T> back) :
				id_(id), shaper_(shaper), Nf_(forward), ginit_(back) {}

		std::string id_;
		SHAPER shaper_;
		TRANSFER_FUNC<T> Nf_;
		BACK_MAP<T> ginit_;

		std::vector<std::string> arg_ids_;
	};

	using summary_series = std::vector<typename iconnector<T>::conn_summary>;

	virtual ~iconnector (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	iconnector<T>* clone (void) const;

	//! move function
	iconnector<T>* move (void);

	//! declare copy assignment to enforce proper g_man_ copy over
	virtual iconnector<T>& operator = (const iconnector<T>& other);

	//! declare move assignment to enforce proper g_man_ copy over
	virtual iconnector<T>& operator = (iconnector<T>&& other);

	// >>>> IDENTIFICATION <<<<
	//! get unique label with arguments
	virtual std::string get_name (void) const;

	// >>>> OBSERVER & OBSERVABLE INFO <<<<
	//! get all observerables
	virtual std::vector<inode<T>*> get_arguments (void) const;

	//! get the number of observables
	virtual size_t n_arguments (void) const;

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! grab a temporary value traversing top-down
	virtual void temporary_eval (const iconnector<T>* target, inode<T>*& out) const = 0;

	//! get forward passing value, (pull data if necessary)
	virtual const tensor<T>* eval (void);

	// >>>> GRAPH STATUS <<<<
	//! summarize this connector
	virtual summary_series summarize (void) const = 0;

	//! check if other connector is in the same graph as this
	bool is_same_graph (const iconnector<T>* other) const;

	//! check if connector n is a potential descendent of this node
	//! will return false negatives if nodes are in a pipeline of a non-variable leaf
	virtual bool potential_descendent (const iconnector<T>* n) const;

	// >>>> NODE STATUS <<<<
	//! add jacobian to the front of the list mapped by leaves
	void set_jacobian_front (JTRANSFER<T> jac, std::vector<variable<T>*> leaves);

	//! add jacobian to the back of the list mapped by leaves
	void set_jacobian_back (JTRANSFER<T> jac, std::vector<variable<T>*> leaves);

	//! freeze or unfreeze the current node
	//! freeze prevents this from updating temporarily instead update is queued to g_man_
	void freeze_status (bool freeze);

protected:
	//! list of jacobian transfer function
	//! to be executed on resulting root node
	//! execution order: top-down
	struct JList
	{
		JList (void) : uid_(nnutils::uuid(this)) {}

		std::string uid_;
		std::list<std::pair<JTRANSFER<T>, inode<T>*> > list_;
	};

	//! graph info shareable between connectors
	struct graph_manager;

	// >>>> CONSTRUCTORS <<<<
	//! Set dependencies
	iconnector (std::vector<inode<T>*> dependencies, std::string label);

	//! Declare copy constructor to enforce proper g_man_ copy over
	iconnector (const iconnector<T>& other);

	//! Declare move constructor to enforce proper g_man_ copy over
	iconnector (iconnector<T>&& other);

	// >>>> MANAGE GRAPH INFO <<<<
	//! Update g_man_ by updating all argument variables
	virtual void update_graph (std::vector<iconnector<T>*> args);

	varptr<T> jacobian_call (varptr<T> out, variable<T>* leaf) const
	{
		auto jpair = this->jacobians_.find(leaf);
		if (this->jacobians_.end() != jpair)
		{
			auto& jlist = jpair->second.list_;
			for (auto it = jlist.rbegin(), et = jlist.rend(); it != et; it++)
			{
				const JTRANSFER<T>& jt = it->first;
				// get the node where jacobian originate from
				const inode<T>* orig = it->second;
				// get origin arguments and its gradients
				std::vector<inode<T>*> args = orig->get_arguments();
				std::vector<inode<T>*> grads(args.size(), nullptr);
				std::transform(args.begin(), args.end(), grads.begin(),
				[this, leaf](inode<T>* arg)
				{
					return this->take_gradient(arg, leaf);
				});
				// operate on out using args and grad
				out = jt(out, args, grads);
			}
		}
		return out;
	}

	//! specialized operator: jacobian operators for each variable,
	//! executed in derive
	std::unordered_map<variable<T>*,JList> jacobians_;

	//! graph meta_data/manager
	graph_manager* g_man_ = nullptr;

private:
	void copy_helper (const iconnector<T>& other)
	{
		jacobians_ = other.jacobians_;
		jacobian_correction(&other);
		// this copies other's dependencies so, this and other share a graph
		if (g_man_) g_man_->suicide(this);
		g_man_ = graph_manager::get(const_cast<iconnector<T>*>(&other), this);
	}

	void move_helper (iconnector<T>&& other)
	{
		jacobians_ = std::move(other.jacobians_);
		jacobian_correction(&other);
		// this copies other's dependencies so, this and other share a graph
		if (g_man_)
		{
			g_man_->suicide(this);
		}
		g_man_ = graph_manager::get(&other, this);
		if (other.g_man_)
		{
			other.g_man_->suicide(&other);
			other.g_man_ = nullptr;
		}
	}

	void jacobian_correction (const inode<T>* other)
	{
		// todo: move this down to immutable,
		// since if mutable, parent can have existing jacobian_ with references to other
		// assert this node has no parent (true when copying immutables)

		// check other's jacobians leafset for references to other and set to this
		for (auto& jpair : jacobians_)
		{
			std::list<std::pair<JTRANSFER<T>,inode<T>*> >& js = jpair.second.list_;
			if (js.back().second == other)
			{
				js.back().second = this;
			}
		}
	}
};

}

#include "../../../src/graph/connector/iconnector.ipp"

#endif /* TENNCOR_ICONNECTOR_HPP */
