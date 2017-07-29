//
//  icompound.cpp
//  cnnet
//
// 	Created by Mingkai Chen on 2017-07-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "compounds/mlp.hpp"

#ifdef ROCNNET_ICOMPOUND_HPP

namespace rocnnet
{

icompound::icompound (std::string scope) : ilayer(scope) {}

icompound::~icompound (void) {}

icompound* icompound::clone (std::string scope) const
{
	return static_cast<icompound*>(this->clone_impl(scope));
}

icompound* icompound::move (void)
{
	return static_cast<icompound*>(this->move_impl());
}

void icompound::initialize (std::string serialname, std::string readscope)
{
	if (readscope.empty()) readscope = scope_;
	std::vector<nnet::variable<double>*> vars = this->get_variables();

	std::vector<nnet::inode<double>*> nv(vars.begin(), vars.end());
	if (nnet::read_inorder<double>(nv, readscope, serialname) && nv.empty())
	{
		return;
	}

	for (nnet::variable<double>* v : vars)
	{
		v->initialize();
	}
}

bool icompound::save (std::string fname, std::string writescope) const
{
	if (writescope.empty()) writescope = scope_;
	std::vector<nnet::variable<double>*> vars = this->get_variables();
	std::vector<nnet::inode<double>*> nv(vars.begin(), vars.end());
	return nnet::write_inorder<double>(nv, writescope, fname);
}

}

#endif