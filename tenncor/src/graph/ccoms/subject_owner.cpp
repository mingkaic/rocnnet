//
//  subject_owner.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-20
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/ccoms/subject_owner.hpp"

#ifdef subject_owner_hpp

namespace ccoms
{

subject_owner::subject_owner (const subject_owner& other)
{
	caller_ = std::make_unique<subject>(*other.caller_, this);
}

subject_owner::subject_owner (void) { caller_ = std::make_unique<subject>(this); }

subject_owner& subject_owner::operator = (const subject_owner& other)
{
	if (&other != this)
	{
		caller_ = std::make_unique<subject>(*other.caller_, this);
	}
	return *this;
}

void subject_owner::notify (subject_owner* grad)
{
	ccoms::update_message msg;
	if (grad)
	{
		msg.grad_ = grad->caller_.get();
	}
	caller_->notify(msg);
}

void subject_owner::notify (ccoms::update_message msg)
{
	caller_->notify(msg);
}

bool subject_owner::no_audience (void) const
{
	return caller_->no_audience();
}

}

#endif