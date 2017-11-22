/*!
 *
 *  identifiable.hpp
 *  cnnet
 *
 *  Purpose:
 *  Interface for obtaining identification information
 *  For when we want to access information at observer/subject level
 *
 *  Created by Mingkai Chen on 2017-11-10
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include <string>

#include "utils/utils.hpp"

#pragma once
#ifndef identifiable_hpp
#define identifiable_hpp

namespace nnet
{

class identifiable
{
public:
	virtual ~identifiable (void) {}

	//! get the unique hash value
	std::string get_uid (void) const
	{
		return id_;
	}

	//! get the non-unique label set by user, denoting node purpose
	virtual std::string get_label (void) const = 0;

private:
	//! uniquely identifier for this node
	const std::string id_ = nnutils::uuid(this);
};

}

#endif /* identifiable_hpp */
