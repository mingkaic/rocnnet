//
//  ilayer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-13.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include <string>
#include <vector>

#include "graph/leaf/variable.hpp"
#include "graph/operations/operations.hpp"

#pragma once
#ifndef ROCNNET_ILAYER_HPP
#define ROCNNET_ILAYER_HPP

namespace rocnnet
{

class ilayer
{
public:
	ilayer (std::string scope);

	virtual ~ilayer (void);

	ilayer* clone (std::string scope = "") const;

	ilayer* move (void);

	ilayer& operator = (const ilayer& other);

	ilayer& operator = (ilayer&& other);

	virtual std::vector<nnet::variable<double>*> get_variables (void) const = 0;

protected:
	ilayer (const ilayer& other, std::string scope);

	ilayer (ilayer&& other);

	virtual ilayer* clone_impl (std::string& scope) const = 0;

	virtual ilayer* move_impl (void) = 0;

	std::string scope_;
};

}

#endif /* ROCNNET_ILAYER_HPP */
