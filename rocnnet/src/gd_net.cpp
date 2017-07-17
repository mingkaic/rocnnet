//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "gd_net.hpp"

#ifdef gd_net_hpp

using namespace nnet;

namespace rocnnet
{

gd_net::gd_net (size_t n_input, std::vector<IN_PAIR> hiddens,
	nnet::gd_updater& updater, std::string scope) :
mlp(n_input, hiddens, scope),
updater_(updater.clone())
{
	size_t n_output = hiddens.back().first;
	train_in_ = new nnet::placeholder<double>(
		std::vector<size_t>{n_input, 0}, "train_in");
	expected_out_ = new nnet::placeholder<double>(
		std::vector<size_t>{n_output, 0}, "expected_out");
	train_setup();
}

gd_net::~gd_net (void)
{
	clean_up();
}

gd_net* gd_net::clone (std::string scope) const
{
	return static_cast<gd_net*>(this->clone_impl(scope));
}

gd_net* gd_net::move (void)
{
	return static_cast<gd_net*>(this->move_impl());
}

gd_net& gd_net::operator = (const gd_net& other)
{
	if (&other != this)
	{
		mlp::operator = (other);
		clean_up();
		copy_helper(other);
	}
	return *this;
}

gd_net& gd_net::operator = (gd_net&& other)
{
	if (&other != this)
	{
		mlp::operator = (std::move(other));
		clean_up();
		move_helper(std::move(other));
	}
	return *this;
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

gd_net::gd_net (const gd_net& other, std::string& scope) :
	mlp(other, scope)
{
	copy_helper(other);
}

gd_net::gd_net (gd_net&& other) :
	mlp(std::move(other))
{
	move_helper(std::move(other));
}

ilayer* gd_net::clone_impl (std::string& scope) const
{
	return new gd_net(*this, scope);
}

ilayer* gd_net::move_impl (void)
{
	return new gd_net(std::move(*this));
}

void gd_net::train_setup (void)
{
	nnet::varptr<double> output = mlp::operator()(train_in_);
	nnet::varptr<double> diff = nnet::varptr<double>(expected_out_) - output;
	nnet::varptr<double> error = diff * diff;
	error_ = static_cast<nnet::iconnector<double>*>(error.get());
	error_->set_label("error");
	updates_ = updater_->calculate(error_);
}

void gd_net::copy_helper (const gd_net& other)
{
	updater_ = other.updater_->clone();
	train_in_ = other.train_in_->clone();
	expected_out_ = other.expected_out_->clone();
	train_setup();
}

void gd_net::move_helper (gd_net&& other)
{
	updater_ = other.updater_->move();
	train_in_ = other.train_in_->move();
	expected_out_ = other.expected_out_->move();
	train_setup();
}

void gd_net::clean_up (void)
{
	if (updater_) delete updater_;
	if (train_in_) delete train_in_;
	if (expected_out_) delete expected_out_;
	updates_.clear();
}

}

#endif /* gd_net_hpp */
