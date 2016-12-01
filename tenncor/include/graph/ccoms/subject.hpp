//
//  subject.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cassert>
#include <unordered_set>
#include "memory/safe_ptr.hpp"

#pragma once
#ifndef subject_hpp
#define subject_hpp

namespace ccoms
{

class iobserver;
class subject;

struct update_message
{
	subject* caller_;
	subject* grad_ = nullptr;
	
	update_message (subject* caller) : caller_(caller) {}
};

// abstract for communication nodes that records leaves
class reactive_node
{
	private:
		std::vector<void**> ptrrs_;
	
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
	
	public:
		virtual ~reactive_node (void)
		{
			for (void** ptrr : ptrrs_)
			{
				*ptrr = nullptr;
			}
		}
		// return true if this is successfully flagged for deletion
		bool safe_destroy (void); // non-virtual to ensure safe virtual inheritance
		// set ptr to null on death
		void set_death (void** ptr)
		{
			ptrrs_.push_back(ptr);
		}
};

// AKA leaf
// subject retains control over all its observers,
// once destroyed, all observers are flagged for deletion

class subject : public reactive_node
{
	private:
		// content (does not own... meaning memory leak/corruption if we delete this without variable)
		nnet::safe_ptr* var_ = nullptr;
		// dependents
		std::unordered_set<iobserver*> audience_;

	protected:
		// must explicitly destroy using delete
		virtual bool suicidal (void) { return false; }
		void attach (iobserver* viewer);
		// if suicidal, safe_destroy when detaching last audience
		virtual void detach (iobserver* viewer);

		friend class iobserver;
		
	public:
		subject (void) {}

		virtual ~subject (void);

		subject (const subject& other) {} // prevent audience from being copied over
	
		void notify (subject* grad = nullptr);
		bool no_audience (void) const;

		// variable connection
		template <typename T>
		void store_var (T* var)
		{
			if (nullptr != var_) delete var_;
			var_ = new nnet::safe_ptr {
				var, // ptr_
				typeid(T), // info
			};
		}

		template <typename T>
		T* to_type (void) {
			return var_->cast<T>();
		}
};

}

#endif /* subject_hpp */
