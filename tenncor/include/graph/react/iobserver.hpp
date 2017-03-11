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

#include <vector>
#include <functional>

#include "subject.hpp"

#pragma once
#ifndef TENNCOR_IOBSERVER_HPP
#define TENNCOR_IOBSERVER_HPP

namespace react
{

class iobserver
{
public:
	//! remove all dependencies
	virtual ~iobserver (void);

	// >>>> COPY && MOVE ASSIGNMENT<<<<
	//! declare copy assignment to copy over dependencies
	virtual iobserver& operator = (const iobserver& other);

	//! declare move assignment to move over dependencies
	virtual iobserver& operator = (iobserver&& other);

	// >>>> MUTATOR <<<<
	//! update observer value according to subject
	//! publicly available to allow explicit updates
	virtual void update (react::subject* arg) = 0;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! default constructor
	iobserver (void);

	//! subscribe to subjects on construction (mostly non-mutable observers)
	iobserver (std::vector<subject*> dependencies);

	// >>>> EXECUTE ON KILL CONDITION <<<<
	//! smart destruction
	virtual void commit_sudoku (void) = 0;

	// >>>> COPY && MOVE CONSTRUCTOR <<<<
	//! copy over dependencies
	iobserver (const iobserver& other);

	//! move over dependencies
	iobserver (iobserver&& other);

	// >>>> DEPENDENCY ACCESS <<<<
	//! access dependency
	void access_dependency (std::function<void(const subject*)> access) const;

	// >>>> DEPENDENCY MANIPULATION <<<<
	//! subscribe: add dependency
	void add_dependency (subject* dep);

	//! unsubscribe: remove dependency
	void remove_dependency (size_t idx);

	//! replace dependency
	void replace_dependency (subject* dep, size_t idx);

private:
	//! update observer value with notification
	void update (size_t dep_idx, notification msg);

	//! copy helper function
	void copy (const iobserver& other);

	//! move helper function
	void move (iobserver& other);

	//! order of subject matters;
	//! observer-subject relation is non-unique
	std::vector<subject*> dependencies_;

	friend class subject;
};

}

#endif /* TENNCOR_IOBSERVER_HPP */
