//
//  subject_owner.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-20
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "subject.hpp"

#pragma once
#ifndef subject_owner_hpp
#define subject_owner_hpp

namespace ccoms
{

class subject_owner
{
	protected:
		// we own this
		std::unique_ptr<subject> caller_;

		subject_owner (const subject_owner& other);
		
		subject_owner (void);

		friend class iobserver;

	public:
		virtual ~subject_owner (void) {}
		
		// COPY
		virtual subject_owner* clone (void) = 0;
		subject_owner& operator = (const subject_owner& other);

		// BRIDGE TO CALLER
		void notify (subject_owner* grad = nullptr);
		void notify (ccoms::update_message msg);
		bool no_audience (void) const;
};

}

#endif /* subject_owner_hpp */