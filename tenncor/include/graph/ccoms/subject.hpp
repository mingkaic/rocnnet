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

// pure abstract / interface for communication nodes that records leaves
class ileaf_handler
{
	protected:
		virtual void merge_leaves (std::unordered_set<ccoms::subject*>& src) = 0;
};

// AKA leaf
// subject retains control over all its observers,
// once destroyed, all observers are destroyed

class subject : public ileaf_handler
{
	private:
		std::unordered_set<iobserver*> audience_;
		
	protected:
		virtual void merge_leaves (std::unordered_set<ccoms::subject*>& src);
		bool no_audience (void);

		friend class iobserver;
		
	public:
		virtual ~subject (void);
	
		void attach (iobserver* viewer);
		virtual void detach (iobserver* viewer);
		void notify (subject* caller = nullptr);
};

}

#endif /* subject_hpp */
