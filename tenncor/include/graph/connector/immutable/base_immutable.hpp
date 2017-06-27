/*!
 *
 *  immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph immutable connector interface
 *  manages tensor data and defines abstract
 *  forward computation and backward computation methods
 *
 *  also defines mergable immutable for a
 *  series of forward and backward passes
 *
 *  Created by Mingkai Chen on 2017-06-26.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/iconnector.hpp"
#include "graph/leaf/variable.hpp"

#pragma once
#ifndef TENNCOR_BASE_IMMUTABLE_HPP
#define TENNCOR_BASE_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
class base_immutable : public iconnector<T>
{
public:
	virtual ~base_immutable (void);

	// >>>> CLONE, COPY && MOVE <<<<
	//! clone function
	base_immutable<T>* clone (void) const;

	//! move function
	base_immutable<T>* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual base_immutable<T>& operator = (const base_immutable<T>& other);

	//! declare move assignment to move over transfer functions
	virtual base_immutable<T>& operator = (base_immutable<T>&& other);

	// >>>> ACCESSORS <<<<
	//! Utility function: get data shape
	virtual tensorshape get_shape (void) const;

	//! Forward passing value
	virtual const tensor<T>* get_eval (void) const;

	//! grab a temporary value traversing top-down
	//! allocates out tensor. caller owns out
	virtual void temporary_eval (const iconnector<T>* target, inode<T>*& out) const;

	//! check if the arguments are good; data is available
	virtual bool good_status (void) const;

	//! get gradient leaves
	virtual void get_leaves (typename inode<T>::GRAD_CACHE& leaves) const;

	// >>>> MUTATORS <<<<
	//! grab operational gradient node, used by other nodes
	//! delay instantiate gcache elements if target leaf was never instantiated
	virtual void get_leaf (varptr<T>& out, variable<T>* leaf);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_leaf
	virtual varptr<T> get_gradient (inode<T>* wrt);

	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (std::vector<size_t> argidx);

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&);

	bool mergible_ = true;

protected:
	//! base_immutable constructing an aggregate transfer function
	base_immutable (std::vector<inode<T>*> args, std::string label);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! declare copy constructor to copy over transfer functions
	base_immutable (const base_immutable<T>& other);

	//! declare move constructor to move over transfer functions
	base_immutable (base_immutable<T>&& other);

	void death_on_broken (void);

	//! forward pass step: populate data_
	virtual void forward_pass (std::vector<size_t>) = 0;

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable<T>* leaf) = 0;

	// >>>> LEAF-GRADIENT CACHE <<<<
	//! maps leaf to gradient node
	//! lazy instantiates gradient nodes
	//! - stores the gradient value wrt each leaf
	//! - record leaf set
	typename inode<T>::GRAD_CACHE gcache_;

	//! inner tensor to cache forward evaluated values
	tensor<T>* data_ = nullptr;

private:
	//! copy helper
	void copy_helper (const base_immutable& other);

	//! move helper
	void move_helper (base_immutable&& other);
};

//! for every node M in root's subgraph, merge M with subjects whose sole audience is M
//! M and merged subjects are deleted
//! (note: varptr counts as an observer, so any subject referenced by varptr by the user will never merge.
//! this is to prevent dangling pointers, since solo_audience_merge is destructive.
//! user's responsible for minimizing the use of varptr)
template <typename T>
void solo_audience_merge (base_immutable<T>*& root);

template <typename T>
class merged_immutable : public base_immutable<T>
{
public:
	//! build a merged_immutable that combines conn and its arguments
	//! ignores arguments indexed by ignore_indices
	static merged_immutable<T>* get (base_immutable<T>* conn,
		std::unordered_set<size_t> ignore_indices = {});

	//! build a merged_immutable that combines conn and specified arguments
	//! ignores arguments indexed by ignore_indices
	static merged_immutable<T>* get (base_immutable<T>* conn,
		std::vector<subject*> args,
		std::unordered_set<size_t> ignore_indices = {});

	// >>>> CLONE, COPY && MOVE <<<<
	//! clone function
	merged_immutable<T>* clone (void) const;

	//! move function
	merged_immutable<T>* move (void);

	virtual std::string get_summaryid (void) const { return summaries_.back().id_; }

	//! summarize this connector
	virtual typename iconnector<T>::summary_series summarize (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! merge conn and its arguments while avoiding arguments specified by indices
	merged_immutable (base_immutable<T>* conn, std::unordered_set<size_t> ignore_indices);

	//! merge conn and specified arguments while avoiding arguments specified by indices
	merged_immutable (base_immutable<T>* conn, std::vector<subject*> args,
		std::unordered_set<size_t> ignore_indices);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

	//! forward pass step: populate data_ (overridden by merged_immutable)
	virtual void forward_pass (std::vector<size_t>);

	//! backward pass step: populate gcache_[leaf] (overridden by merged_immutable)
	virtual void backward_pass (variable<T>* leaf);

private:
	void init_helper (typename iconnector<T>::summary_series top_summary,
		std::vector<subject*> args, std::unordered_set<size_t> ignore_indices);

	template <typename U>
	U summary_traversal (std::unordered_map<std::string,U> arg_map,
		std::function<U(std::vector<U>,typename iconnector<T>::conn_summary&)> op);

	struct temp_immutable;

	std::unordered_map<std::string,std::vector<T> > raw_intermediates_;

	//! an array of pointers to raw data either in raw_intermediates_ vector or from argument tensors
	//! intentionally kept ambiguous to avoid computation overhead during forward_pass
	//! populated once
	std::unordered_map<std::string,std::vector<const T*> > arg_ptrs_;

	typename iconnector<T>::summary_series summaries_;
};

}

#include "../../../../src/graph/connector/immutable/base_immutable.ipp"

#endif /* TENNCOR_BASE_IMMUTABLE_HPP */
