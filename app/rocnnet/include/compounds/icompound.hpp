//
//  icompound.hpp
//  cnnet
//
// 	Created by Mingkai Chen on 2017-07-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "memory/tensor_io.hpp"
#include "utils/futils.hpp"

#include "layers/fc_layer.hpp"

#pragma once
#ifndef ROCNNET_ICOMPOUND_HPP
#define ROCNNET_ICOMPOUND_HPP

namespace rocnnet
{

using VAR_FUNC = std::function<nnet::varptr<double>(nnet::inode<double>*)>;
using IN_PAIR = std::pair<size_t, VAR_FUNC>;

class icompound : public ilayer
{
public:
	// trust that passed in operations are unconnected
	icompound (std::string scope);

	virtual ~icompound (void);

	icompound* clone (std::string scope = "") const;

	icompound* move (void);

	virtual nnet::varptr<double> prop_up (nnet::inode<double>* input) = 0;

	void initialize (std::string serialname = "", std::string readscope = "");

	bool save (std::string fname, std::string writescope = "") const;

	virtual size_t get_ninput (void) const = 0;

	virtual size_t get_noutput (void) const = 0;

protected:
	icompound (const icompound& other, std::string& scope) : ilayer(other, scope) {}

	icompound (icompound&& other) : ilayer(std::move(other)) {}
};

}

#endif /* ROCNNET_ICOMPOUND_HPP */
