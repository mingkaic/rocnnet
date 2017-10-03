//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "models/gd_net.hpp"

#ifdef ROCNNET_GDN_HPP

using namespace nnet;

namespace rocnnet
{

gd_net::gd_net (icompound* brain, nnet::gd_updater& updater, std::string scope) :
	brain_(brain),
	updater_(updater.clone()),
	scope_(scope)
{
	test_in_ = new placeholder<double>(
		std::vector<size_t>{brain->get_ninput(), 0}, scope + "_in");
	train_in_ = new nnet::placeholder<double>(
		std::vector<size_t>{brain->get_ninput(), 0}, "train_in");
	expected_out_ = new nnet::placeholder<double>(
		std::vector<size_t>{brain->get_noutput(), 0}, "expected_out");
	setup();
}

gd_net::~gd_net (void)
{
	clean_up();
}

gd_net::gd_net (const gd_net& other, std::string scope)
{
	copy_helper(other, scope);
}

gd_net::gd_net (gd_net&& other)
{
	move_helper(std::move(other));
}

gd_net& gd_net::operator = (const gd_net& other)
{
	if (&other != this)
	{
		clean_up();
		copy_helper(other, "");
	}
	return *this;
}

gd_net& gd_net::operator = (gd_net&& other)
{
	if (&other != this)
	{
		clean_up();
		move_helper(std::move(other));
	}
	return *this;
}

std::vector<double> gd_net::operator () (std::vector<double>& input)
{
	*test_in_ = input;
	return nnet::expose<double>(test_out_);
}

// batch gradient descent
// 1/m*sum_m(Err(X, Y)) once matrix operations support auto derivation
// then apply cost funct to grad desc alg:
// new weight = old weight - learning_rate * cost func gradient over old weight
// same thing with bias (should experience no rocnnet decrease due to short circuiting)
void gd_net::train (std::vector<double>& train_in, std::vector<double>& expected_out)
{
	*train_in_ = train_in;
	*expected_out_ = expected_out;
	error_->freeze_status(true); // freeze
	for (auto& trainer : updates_)
	{
		trainer(true);
	}
	error_->freeze_status(false); // update again
}

void gd_net::initialize (std::string serialname, std::string readscope)
{
	if (readscope.empty()) readscope = scope_;
	brain_->initialize(serialname, readscope);
}

bool gd_net::save (std::string fname, std::string writescope) const
{
	if (writescope.empty()) writescope = scope_;
	return brain_->save(fname, writescope);
}

void gd_net::setup (void)
{
	test_out_ = brain_->prop_up(test_in_);
	nnet::varptr<double> output = brain_->prop_up(train_in_);
	nnet::varptr<double> diff = nnet::varptr<double>(expected_out_) - output;
	nnet::varptr<double> error = diff * diff;
	error_ = static_cast<nnet::iconnector<double>*>(error.get());
	error_->set_label("error");
	updates_ = updater_->calculate(error_);
}

void gd_net::copy_helper (const gd_net& other, std::string scope)
{
	scope_ = other.scope_;
	updater_ = other.updater_->clone();
	test_in_ = other.test_in_->clone();
	train_in_ = other.train_in_->clone();
	expected_out_ = other.expected_out_->clone();
	brain_ = other.brain_->clone(scope);
	setup();
}

void gd_net::move_helper (gd_net&& other)
{
	scope_ = std::move(other.scope_);
	updater_ = other.updater_->move();
	test_in_ = other.test_in_->move();
	test_out_ = other.test_out_->move();
	train_in_ = other.train_in_->move();
	expected_out_ = other.expected_out_->move();
	brain_ = other.brain_->move();
	setup();
}

void gd_net::clean_up (void)
{
	if (test_in_) delete test_in_;
	if (train_in_) delete train_in_;
	if (expected_out_) delete expected_out_;
	if (updater_) delete updater_;
	if (brain_) delete brain_;
	test_in_ = nullptr;
	test_out_ = nullptr;
	train_in_ = nullptr;
	expected_out_ = nullptr;
	updater_ = nullptr;
	brain_ = nullptr;
	updates_.clear();
}

}

#endif
