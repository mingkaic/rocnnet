/*!
 *
 *  inode.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph variable interface
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "tensor/tensor_handler.hpp"
#include "tensor/tensor.hpp"
#include "graph/react/subject.hpp"
#include "graph/react/iobserver.hpp"

#pragma once
#ifndef TENNCOR_INODE_HPP
#define TENNCOR_INODE_HPP

namespace nnet
{

template <typename T>
class varptr;

template <typename T>
class ileaf;

template <typename T>
class variable;

template <typename T>
class inode : public subject
{
public:
	virtual ~inode (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	inode<T>* clone (void) const;

	//! move function
	inode<T>* move (void);

	//! declare copy assignment to prevent id_ copy over
	virtual inode<T>& operator = (const inode<T>& other);

	//! declare move assignment to prevent id_ copy over
	virtual inode<T>& operator = (inode<T>&& other);

	// >>>> IDENTIFICATION <<<<
	//! get the non-unique label set by user, denoting node purpose
	virtual std::string get_label (void) const;

	//! get beautified summary of name and uid, structure varies for inheritors
	virtual std::string get_name (void) const;

	//! get name described in summary, defaults to name, may differ for special nodes
	virtual std::string get_summaryid (void) const;

	//! modify label to better describe node purpose
	void set_label (std::string label);

	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const = 0;

	//>>>> OBSERVER & OBSERVABLE INFO <<<<
	//! get all observerables
	virtual std::vector<inode<T>*> get_arguments (void) const = 0;

	//! get the number of observables
	virtual size_t n_arguments (void) const = 0;

	//! get all possible observers with specified label
	//! return true if such observer is found
	bool find_audience (std::string label, std::unordered_set<inode<T>*>& audience) const;

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! get forward passing value, (pull data if necessary)
	virtual const tensor<T>* eval (void) = 0;

	//! get top-level gradient value, used by root nodes
	virtual varptr<T> derive (inode<T>* wrt) = 0;

	//! utility function: get forward data shape
	virtual tensorshape get_shape (void) const = 0;

	// >>>> GRAPH STATUS <<<<
	//! merge/update the gradient/leaf info
	virtual std::unordered_set<ileaf<T>*> get_leaves (void) const = 0;

	// >>>> NODE STATUS <<<<
	//! check if data is available
	virtual bool good_status (void) const = 0;

	//! read tensor data from protobuf, may modify current data to provide best information
	//! return true if data is stored successfully
	virtual bool read_proto (const tenncor::tensor_proto& proto) = 0;

	//! store any special-case numerical data using a string key
	void set_metadata (std::string key, size_t value);

	//! propagate special-case data from specified node to this node
	void extract_metadata (inode<T>* n);

	//! check for special-case numerical data
	optional<size_t> get_metadata (std::string key) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! default constructor
	inode (std::string name);

	//! declare copy constructor to prevent id_ copy over
	inode (const inode<T>& other);

	//! declare move constructor to prevent id_ copy over
	inode (inode<T>&& other);

	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone abstraction function
	virtual inode<T>* clone_impl (void) const = 0;

	//! move abstraction function
	virtual inode<T>* move_impl (void) = 0;

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! get forward passing value
	virtual const tensor<T>* get_eval (void) const = 0;

	//! grab operational gradient node, used by other nodes
	//! adds to internal caches if need be
	virtual inode<T>* get_gradient (variable<T>* leaf) = 0;

	//! obtain tensor data from source
	const tensor<T>* take_eval (inode<T>* source) const;

	//! allow inheritants to access source's get_gradient with parameter leaf
	inode<T>* take_gradient (inode<T>* source, variable<T>* leaf) const;

private:
	//! describes this node's purpose
	std::string label_;

	//! record special-case numerical data
	std::unordered_map<std::string, size_t> metadata_;
};

//! helper function for exposing node's data (alternatively: node::get_eval()->expose())
template <typename T>
std::vector<T> expose (inode<T>* var);

}

#include "../../src/graph/inode.ipp"

#endif /* TENNCOR_INODE_HPP */
