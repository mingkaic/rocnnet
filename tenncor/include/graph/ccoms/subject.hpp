//
//  subject.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <unordered_set>

#pragma once
#ifndef subject_hpp
#define subject_hpp

namespace ccoms
{

class iobserver;
class subject;

// abstract for communication nodes that records leaves
class reactive_node
{
	protected:
		// returns true if suicide on safe_destroy
		// we should always have protected constructors with a static builder
		// if suicidal is true, since suicide never accounts for stack allocation
		virtual bool suicidal (void) = 0;
		virtual void merge_leaves (std::unordered_set<subject*>& src) = 0;
		template <typename E, typename ...A>
		
		// allocation is moved here because I want safe destroy and node_allocation 
		// to eventually form a separate abstract factory object
		static E* node_allocate (A... args)
		{
			// memory management for nodes
			// default new on heap for now, could be machine dependent later
			return new E(args...);
		}
	
	public:
		virtual ~reactive_node (void) {}
		// return true if this is successfully flagged for deletion
		bool safe_destroy (void); // non-virtual to ensure safe virtual inheritance
};

// AKA leaf
// subject retains control over all its observers,
// once destroyed, all observers are flagged for deletion

class subject : public reactive_node
{
	private:
		std::unordered_set<iobserver*> audience_;

	protected:
		virtual void merge_leaves (std::unordered_set<subject*>& src);
		bool no_audience (void);
		
		// must explicitly destroy using delete
		virtual bool suicidal (void) { return false; }
		void attach (iobserver* viewer);
		// if suicidal, safe_destroy when detaching last audience
		void detach (iobserver* viewer);

		friend class iobserver;
		
	public:
		virtual ~subject (void);
	
		void notify (subject* caller = nullptr);
};

}

#endif /* subject_hpp */
