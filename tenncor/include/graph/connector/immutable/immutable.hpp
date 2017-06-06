/*!
 *
 *  immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph immutable connector interface
 *  manages connector gradient information
 *
 *  Created by Mingkai Chen on 2017-02-28.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/iconnector.hpp"
#include "graph/leaf/variable.hpp"

#pragma once
#ifndef TENNCOR_IMMUTABLE_HPP
#define TENNCOR_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
class merged_immutable;

template <typename T>
class immutable : public iconnector<T>
{
public:
	//! builder for immutables
	static immutable<T>* get (std::vector<inode<T>*> args,
		SHAPER shaper, FORWARD_OP<T> Nf, BACK_MAP<T> F,
		std::string name, inode<T>* ignore_jacobian = nullptr);

	//! destructor
	virtual ~immutable (void);

	// >>>> CLONE, COPY && MOVE <<<<
	//! clone function
	immutable<T>* clone (void) const;

	//! move function
	immutable<T>* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual immutable<T>& operator = (const immutable<T>& other);

	//! declare move assignment to move over transfer functions
	virtual immutable<T>& operator = (immutable<T>&& other);

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
	virtual void get_leaves (
		typename inode<T>::GRAD_CACHE& leaves) const;

	// >>>> MUTATORS <<<<
	//! grab operational gradient node, used by other nodes
	//! delay instantiate gcache elements if target leaf was never instantiated
	virtual void get_leaf (inode<T>*& out, variable<T>* leaf);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_leaf
	virtual varptr<T> get_gradient (inode<T>* wrt);

	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (subject* arg);

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&);

	//! summarize this immutable
	virtual std::vector<typename iconnector<T>::conn_summary> summarize (void) const;

	bool mergible_ = true;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! mutable constructor defining transfer functions
	immutable (std::vector<inode<T>*> args,
		SHAPER shaper, FORWARD_OP<T> Nf,
		BACK_MAP<T> F, std::string label);

	immutable (std::vector<inode<T>*> args,
		typename iconnector<T>::conn_summary s) :
	iconnector<T>(args, s.id_),
	Nf_(s.Nf_), ginit_(s.ginit_) { update(nullptr); }

	//! copy everything but with new arguments
	immutable (std::vector<inode<T>*> args, const immutable<T>& other);

	// >>>> EXECUTE ON KILL CONDITION <<<<
	//! ovride smart destruction,
	//! executed when any dependency is destroyed
	virtual void death_on_broken (void);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! implement clone function
	virtual inode<T>* clone_impl (void) const;

	//! move implementation
	virtual inode<T>* move_impl (void);

	//! declare copy constructor to copy over transfer functions
	immutable (const immutable<T>& other);

	//! declare move constructor to move over transfer functions
	immutable (immutable<T>&& other);

	//! forward pass step: populate data_ (overridden by merged_immutable)
	virtual void forward_pass (std::vector<const tensor<T>*> tens);

	//! backward pass step: populate gcache_[leaf] (overridden by merged_immutable)
	virtual void backward_pass (std::vector<inode<T>*> deps, variable<T>* leaf);

	// >>>> LEAF-GRADIENT CACHE <<<<
	//! maps leaf to gradient node
	//! lazy instantiates gradient nodes
	//! - stores the gradient value wrt each leaf
	//! - record leaf set
	typename inode<T>::GRAD_CACHE gcache_;

private:
	//! copy helper
	void copy_helper (const immutable& other);

	//! move helper
	void move_helper (immutable&& other);

	// >>>> FORWARD OPERATIONS <<<<
	//! inner tensor to cache forward evaluated values
	tensor<T>* data_ = nullptr;

	//! forward transfer function
	transfer_func<T> Nf_; //! calculates forward passing data

	// >>>> GRAD_ INITIALIZER <<<<
	//! backward transfer function to
	//! lazy instantiate gradient cache values
	BACK_MAP<T> ginit_;
};

//! for every graph in the subgraph, destructively merge nodes with only a single audience
template <typename T>
void solo_merge (immutable<T>*& root);

template <typename T>
class merged_immutable : public immutable<T>
{
public:
	//! builders for merged_immutables
	static merged_immutable<T>* get (immutable<T>* conn, bool destructive = true);
	
	static merged_immutable<T>* get (immutable<T>* conn, std::unordered_set<size_t> ignore_indices)
	{
		return new merged_immutable<T>(conn, ignore_indices);
	}

	// >>>> CLONE, COPY && MOVE <<<<
	//! clone function
	merged_immutable<T>* clone (void) const;

	//! move function
	merged_immutable<T>* move (void);

	//! summarize this connector
	virtual std::vector<typename iconnector<T>::conn_summary> summarize (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! merged_immutable constructor merging connector and its children and destroys conn
	merged_immutable (immutable<T>* conn);

	//! non-destructive merge connector with its dependencies
	//! avoid merging dependencies specified by indices
	//! non-destructive cannot copy over audience, 
	//! (audience copy forces a deep copy for every super consumer of this node)
	merged_immutable (immutable<T>* conn, std::unordered_set<size_t> ignore_indices);

	// todo: replace with better alternative
	//! temporary constructor for backprop only: never destructive
	merged_immutable (std::vector<inode<T>*> args,
		typename iconnector<T>::conn_summary s) :
	immutable<T>(args, s)
	{
		this->set_label("merge_temp"+s.id_);
		for (size_t i = 0, n = args.size(); i < n; i++)
			sub_mapper_.push_back({"", i});
	}

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
	//! refresh summaries and sub_mapper from current dependencies and input sub_mapper
	std::vector<subject*> summary_merge (
		std::vector<std::pair<std::string,size_t> > othersubmapper, 
		std::unordered_set<size_t> ignore_argidx = {});

	std::vector<typename iconnector<T>::conn_summary> summaries_;

	//! map dependencies to the id of the summary that consumes it
	std::vector<std::pair<std::string,size_t> > sub_mapper_;
};

}

#include "../../../../src/graph/connector/immutable/immutable.ipp"

#endif /* TENNCOR_IMMUTABLE_HPP */
