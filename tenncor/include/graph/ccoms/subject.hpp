//
//  subject.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <unordered_set>
#include <memory>

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
		virtual bool suicidal (void) = 0;
		virtual void merge_leaves (std::unordered_set<ccoms::subject*>& src) = 0;
	
	public:
		// return true if this is successfully flagged for deletion
		bool safe_destroy (void);
};

// AKA leaf
// subject retains control over all its observers,
// once destroyed, all observers are flagged for deletion

class subject : public reactive_node
{
	private:
		std::unordered_set<iobserver*> audience_;

	protected:
		virtual void merge_leaves (std::unordered_set<ccoms::subject*>& src);
		bool no_audience (void);
		
		// must explicitly destroy using delete
		virtual bool suicidal (void) { return false; }

		friend class iobserver;
		
	public:
		virtual ~subject (void);
	
		void attach (iobserver* viewer);
		// if suicidal, safe_destroy when detaching last audience
		void detach (iobserver* viewer);
		void notify (subject* caller = nullptr);
};

}

#endif /* subject_hpp */
