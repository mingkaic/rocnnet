/*!
 *
 *  base_immutable.hpp
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
	//! type for mapping leaf nodes to derivative with respect to leaf
	using GRAD_CACHE = std::unordered_map<ileaf<T>*,varptr<T> >;

	virtual ~base_immutable (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	base_immutable<T>* clone (void) const;

	//! move function
	base_immutable<T>* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual base_immutable<T>& operator = (const base_immutable<T>& other);

	//! declare move assignment to move over transfer functions
	virtual base_immutable<T>& operator = (base_immutable<T>&& other);

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
	virtual std::unordered_set<ileaf<T>*> get_leaves (void) const;

	// >>>> NODE STATUS <<<<
	//! check if the arguments are good; data is available
	virtual bool good_status (void) const;

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&);

	//! public flag notifying whether this node can be merged
	bool mergible_ = true;

	// >>>> CALLED BY OBSERVER TO UPDATE <<<<
	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (std::unordered_set<size_t> argidx);

	// >>>> TODO: HIDE THIS <<<<

protected:
	// >>>> CONSTRUCTORS <<<<
	//! base_immutable constructing an aggregate transfer function
	base_immutable (std::vector<inode<T>*> args, std::string label);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! declare copy constructor to copy over transfer functions
	base_immutable (const base_immutable<T>& other);

	//! declare move constructor to move over transfer functions
	base_immutable (base_immutable<T>&& other);

	// >>>> KILL CONDITION <<<<
	//! suicides when all observ
	void death_on_broken (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! Forward passing value
	virtual const tensor<T>* get_eval (void) const;

	//! grab operational gradient node, used by other nodes
	//! delay instantiate gcache elements if target leaf was never instantiated
	virtual inode<T>* get_gradient (variable<T>* leaf);

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_
	virtual void forward_pass (void) = 0;

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable<T>* leaf) = 0;

	//! maps leaf to gradient node
	//! lazy instantiates gradient nodes
	//! - stores the gradient value wrt each leaf
	//! - record leaf set
	typename base_immutable<T>::GRAD_CACHE gcache_;

// todo: have an option to disable data_ caching for performance boost
	//! inner tensor to cache forward evaluated values
	tensor<T>* data_ = nullptr;

private:
	//! copy helper
	void copy_helper (const base_immutable& other);

	//! move helper
	void move_helper (base_immutable&& other);

	//! backward_evaluation helper
	void eval_helper (const iconnector<T>* target, inode<T>*& out) const;
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
	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! build a merged_immutable that combines conn and its arguments
	//! ignores arguments indexed by ignore_indices
	static merged_immutable<T>* get (base_immutable<T>* conn,
		std::unordered_set<size_t> ignore_indices = {},
		bool disabled_update = false);

	//! build a merged_immutable that combines conn and specified arguments
	//! ignores arguments indexed by ignore_indices
	static merged_immutable<T>* get (base_immutable<T>* conn,
		std::vector<subject*> args,
		std::unordered_set<size_t> ignore_indices = {},
		bool disabled_update = false);

	// >>>> CLONER & MOVER <<<<
	//! clone function
	merged_immutable<T>* clone (void) const;

	//! move function
	merged_immutable<T>* move (void);

	// >>>> IDENTIFICATION <<<<
	//! get name described in summary, defaults to name, may differ for special nodes
	virtual std::string get_summaryid (void) const;

	// >>>> GRAPH STATUS <<<<
	//! summarize this connector
	virtual typename iconnector<T>::summary_series summarize (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! merge conn and its arguments while avoiding arguments specified by indices
	merged_immutable (base_immutable<T>* conn,
		std::unordered_set<size_t> ignore_indices, bool disabled_update);

	//! merge conn and specified arguments while avoiding arguments specified by indices
	merged_immutable (base_immutable<T>* conn, std::vector<subject*> args,
		std::unordered_set<size_t> ignore_indices, bool disabled_update);

	// >>>> POLYMORPHIC CLONERS <<<<
	//! implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_ (overridden by merged_immutable)
	virtual void forward_pass (void);

	//! backward pass step: populate gcache_[leaf] (overridden by merged_immutable)
	virtual void backward_pass (variable<T>* leaf);

private:
	struct temp_immutable;

	//! constructor helper
	void init_helper (typename iconnector<T>::summary_series top_summary,
		std::vector<subject*> args, std::unordered_set<size_t> ignore_indices);

	//! forward and backward helper
	//! op determines how U data is evaluated
	template <typename U>
	U summary_traversal (std::unordered_map<std::string,U> arg_map,
		std::function<U(std::vector<U>,typename iconnector<T>::conn_summary&)> op);

	//! summary.id_ -> tensor::raw_data_ equivalent vector
	std::unordered_map<std::string,std::vector<T> > raw_intermediates_;

	//! an array of pointers to raw data either in raw_intermediates_ vector or from argument tensors
	//! intentionally kept ambiguous to avoid computation overhead during forward_pass
	//! populated once
	std::unordered_map<std::string,std::vector<const T*> > arg_ptrs_;

	//! vector of summaries (of previously merged nodes)
	//! describing how to forward eval and build backward node (derive)
	typename iconnector<T>::summary_series summaries_;
};

}

#include "../../../../src/graph/connector/immutable/base_immutable.ipp"

#endif /* TENNCOR_BASE_IMMUTABLE_HPP */
