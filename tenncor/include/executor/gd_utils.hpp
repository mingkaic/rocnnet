//
// Created by Mingkai Chen on 2017-04-27.
//

#include "graph/varptr.hpp"
#include "graph/leaf/variable.hpp"
#include "graph/connector/immutable/elementary.hpp"

#pragma once
#ifndef ROCNNET_GD_UTILS_HPP
#define ROCNNET_GD_UTILS_HPP


namespace nnet
{

//! gradient descent algorithm interface
template <typename T>
struct gd_updater
{
	virtual ~gd_updater(void) {}

	gd_updater<T>* clone (void) { return clone_impl(); }

	gd_updater<T>* move (void) { return move_impl(); }

	virtual std::vector<nnet::variable_updater<T> > calculate (inode<T>* root) = 0;

	double learning_rate_ = 0.5;

protected:
	virtual gd_updater<T>* clone_impl (void) = 0;

	virtual gd_updater<T>* move_impl (void) = 0;
};

//! batch gradient descent algorithm
struct bgd_utils : gd_updater<double>
{
	bgd_utils* clone (void) { return static_cast<bgd_utils*>(clone_impl()); }

	bgd_utils* move (void) { return static_cast<bgd_utils*>(move_impl()); }

	virtual std::vector<nnet::variable_updater<double> > calculate (inode<double>* root);

protected:
	virtual gd_updater<double>* clone_impl (void)
	{
		return new bgd_utils(*this);
	}

	virtual gd_updater<double>* move_impl (void)
	{
		return new bgd_utils(std::move(*this));
	}
};


}


#endif /* ROCNNET_GD_UTILS_HPP */
