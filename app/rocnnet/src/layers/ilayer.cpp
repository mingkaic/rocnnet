//
//  ilayer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-13.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "layers/ilayer.hpp"

#ifdef ROCNNET_ILAYER_HPP

namespace rocnnet
{

ilayer::ilayer (std::string scope) : scope_(scope) {}

ilayer::~ilayer (void) {}

ilayer* ilayer::clone (std::string scope) const
{
	return this->clone_impl(scope);
}

ilayer* ilayer::move (void)
{
	return this->move_impl();
}

ilayer& ilayer::operator = (const ilayer& other)
{
	if (&other != this)
	{
		scope_ = other.scope_;
	}
	return *this;
}

ilayer& ilayer::operator = (ilayer&& other)
{
	if (&other != this)
	{
		scope_ = std::move(other.scope_);
	}
	return *this;
}

ilayer::ilayer (const ilayer& other, std::string scope)
{
	if (scope.empty())
	{
		scope_ = other.scope_ + "_cpy";
	}
	else
	{
		scope_ = scope;
	}
}

ilayer::ilayer (ilayer&& other) :
	scope_(std::move(other.scope_)) {}

}

#endif
