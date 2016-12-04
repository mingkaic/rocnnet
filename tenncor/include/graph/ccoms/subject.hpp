//
//  subject.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cassert>
#include <unordered_set>
#include "utils.hpp"

#pragma once
#ifndef subject_hpp
#define subject_hpp

namespace ccoms
{

class iobserver;
class subject;
class subject_owner;

template <typename T>
class igraph;

struct update_message
{
	subject* grad_ = nullptr; // this should be reset to null after every update to prevent propogating up the graph (TODO: consider moving to caller)
	iobserver* jacobi_ = nullptr;
	bool leave_update_ = false;
};

struct caller_info
{
	subject* caller_;
	size_t caller_idx_ = 0;

	caller_info (subject* caller = nullptr) : caller_(caller) {}
};

// abstract for communication nodes that records leaves
class reactive_node
{
	private:
		// used exclusively for controlling to pointers of this
		// upon destruction all pointers ptrrs point to will be set to null
		std::unordered_set<void**> ptrrs_;
	
	protected:
		// returns true if suicide on safe_destroy
		// we should always have protected constructors with a static builder
		// if suicidal is true, since suicide never accounts for stack allocation
		virtual bool suicidal (void) = 0;

		// allocation is moved here because I want safe destroy and node_allocation 
		// to eventually form a separate abstract factory object
		template <typename E, typename ...A>
		static E* node_allocate (A... args)
		{
			// memory management for nodes
			// default new on heap for now, could be machine dependent later
			return new E(args...);
		}

		reactive_node (void) {}
		reactive_node (reactive_node& other) {} // prevent ptrrs from being copied
	
	public:
		virtual ~reactive_node (void); // set all ptrrs' pointer to null
		// return true if this is successfully flagged for deletion
		virtual bool safe_destroy (void);
		// set ptr to null on death,
		// ptr might not necessary point to this, ptr could point to something affecting this
		// this distinction must be determined by the caller, be warned
		void set_death (void** ptr);
		void unset_death (void** ptr);
};

// subject retains control over all its observers,
// once destroyed, all observers are flagged for deletion

class subject : public reactive_node
{
	private:
		// content (does not own... meaning memory leak/corruption if we delete this without variable; will occur for constant)
		subject_owner* var_ = nullptr;
		// dependents
		std::unordered_map<iobserver*, std::vector<size_t> > audience_;

	protected:
		// must explicitly destroy using delete
		virtual bool suicidal (void) { return false; }
		void attach (iobserver* viewer, size_t idx);
		// if suicidal, safe_destroy when detaching last audience
		virtual void detach (iobserver* viewer);

		friend class iobserver;
		
	public:
		subject (subject_owner* owner);
		virtual ~subject (void);
		// prevent audience from being copied over
		subject (const subject& other, subject_owner* owner);
	
		void notify (update_message msg = update_message());
		bool no_audience (void) const;
		// override reactive_node's safe_destroy to kill var_ should subject become suicidal
		// killing var_ is essentially the same as suicide, since var_ will in turn kill this
		virtual bool safe_destroy (void);
		subject_owner* get_owner (void);
};

class subject_owner
{
	protected:
		subject* caller_ = nullptr;

		void copy (const subject_owner& other);
		subject_owner (const subject_owner& other);
		subject_owner (void);

	public:
		virtual ~subject_owner (void);

		// BRIDGE TO CALLER
		void notify (subject_owner* grad = nullptr);
		void notify (ccoms::update_message msg);
		bool no_audience (void) const;
};

}

#endif /* subject_hpp */
