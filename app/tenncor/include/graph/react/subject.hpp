/*!
 *
 *  subject.hpp
 *  cnnet
 *
 *  Purpose:
 *  subject notifies observers when changes occur
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "utils/utils.hpp"

#pragma once
#ifndef TENNCOR_SUBJECT_HPP
#define TENNCOR_SUBJECT_HPP

#include <unordered_map>
#include <unordered_set>
#include <experimental/optional>

using namespace std::experimental;

namespace nnet
{

class iobserver;

//! notification messages
enum notification
{
	UNSUBSCRIBE,
	UPDATE
};

//! subject retains control over all its observers,
//! once destroyed, all observers are flagged for deletion
class subject
{
public:
	virtual ~subject (void);

	// >>>> ASSIGNMENT OPERATORS <<<<
	//! declare copy assignment to prevent audience_ copy over
	virtual subject& operator = (const subject& other);

	//! declare move assignment since copy is declared
	virtual subject& operator = (subject&& other);

	// >>>> CALL OBSERVERS <<<<
	//! notify audience of subject update
	void notify (notification msg) const;

	// >>>> OBSERVER INFO <<<<
	//! determine whether the subject has an audience
	bool no_audience (void) const;

	size_t n_audience (void) const;

	//! replace other with this instance for all parents of other
	void steal_observers (subject* other);

protected:
	//! explicit default constructor to allow copy and move constructors
	subject (void);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! Declare copy constructor to prevent audience from being copied over
	subject (const subject& other);

	//! Declare move constructor since copy is declared
	subject (subject&& other);

	// >>>> KILL CONDITION <<<<
	//! smart destruction: called when lacking observables
	//! action: nothing, subjects do not die by default
	virtual void death_on_noparent (void);

	// >>>> OBSERVER MUTATORS SHARED WITH OBSERVERS <<<<
	//! Add observer to audience
	void attach (iobserver* viewer, size_t idx);

	//! Remove observer from audience
	virtual void detach (iobserver* viewer);

	//! Remove observer-index data from audience
	virtual void detach (iobserver* viewer, size_t idx);

	friend class iobserver;

	// >>>> OBSERVERS DATA SHARED WITH INHERITORS <<<
	//! observers -> { subject index in observer referencing this }
	std::unordered_map<iobserver*, std::unordered_set<size_t> > audience_;
};

}

#endif /* TENNCOR_SUBJECT_HPP */
