/*!
 *
 *  iobserver.hpp
 *  cnnet
 *
 *  Purpose:
 *  observer interface is notified
 *  by subjects when changes occur
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "subject.hpp"

#pragma once
#ifndef TENNCOR_IOBSERVER_HPP
#define TENNCOR_IOBSERVER_HPP

#include <vector>
#include <functional>
#include <algorithm>

namespace nnet
{

class iobserver
{
public:
	virtual ~iobserver (void);

	// >>>> ASSIGNMENT OPERATORS <<<<
	//! declare copy assignment to copy over dependencies
	virtual iobserver& operator = (const iobserver& other);

	//! declare move assignment to move over dependencies
	virtual iobserver& operator = (iobserver&& other);

	// >>>> OBSERVER INFO <<<<
	//! determine whether this observes sub
	bool has_subject (subject* sub) const;

	// >>>> CALLED BY OBSERVER TO UPDATE <<<<
	//! update observer value according to subject
	//! publicly available to allow explicit updates
	virtual void update (std::unordered_set<size_t> argidx) = 0;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! default constructor
	iobserver (void);

	//! subscribe to subjects on construction (mostly non-mutable observers)
	iobserver (std::vector<subject*> dependencies);

	//! copy over dependencies
	iobserver (const iobserver& other);

	//! move over dependencies
	iobserver (iobserver&& other);

	// >>>> KILL CONDITION <<<<
	//! smart destruction: call when any observer is broken
	virtual void death_on_broken (void) = 0;

	// >>>> DEPENDENCY MUTATORS <<<<
	//! subscribe: add dependency
	void add_dependency (subject* dep);

	//! unsubscribe: remove dependency
	void remove_dependency (size_t idx);

	//! replace dependency
	void replace_dependency (subject* dep, size_t idx);

	//! order of subject matters;
	//! observer-subject relation is non-unique
	std::vector<subject*> dependencies_;

private:
	// >>>> NOTIFICATION MESSAGE MANAGER <<<
	//! update observer value with notification
	virtual void update (std::unordered_set<size_t> dep_indices, notification msg);

	//! copy helper function
	void copy_helper (const iobserver& other);

	//! move helper function
	void move_helper (iobserver&& other);

	friend class subject;
};

}

#endif /* TENNCOR_IOBSERVER_HPP */
