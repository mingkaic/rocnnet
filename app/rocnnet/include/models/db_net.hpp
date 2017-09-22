//
//  db_net.hpp
//  cnnet
//
//	Implements Restricted Boltzmann Machine
//	Setup cost and training follows implemented from http://deeplearning.net/tutorial/code/DBN.py
//
//  Created by Mingkai Chen on 2017-07-26.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "models/rbm.hpp"

#ifndef ROCNNET_DB_NET_HPP
#define ROCNNET_DB_NET_HPP

namespace rocnnet
{

using pretrain_t = std::vector<std::pair<nnet::variable_updater<double>, nnet::varptr<double> > >;

class db_net : public icompound
{
public:
	db_net (size_t n_input, std::vector<size_t> hiddens,
		std::string scope = "DBN");

	virtual ~db_net (void);

	db_net* clone (std::string scope = "") const;

	db_net* move (void);

	db_net& operator = (const db_net& other);

	db_net& operator = (db_net&& other);

	nnet::varptr<double> operator () (nnet::placeholder<double>& input);

	pretrain_t pretraining_functions (
		nnet::placeholder<double>& input,
		double learning_rate = 1e-3,
		size_t n_cont_div = 10);

	update_cost_t build_finetune_functions (
		nnet::placeholder<double>& train_in,
		nnet::placeholder<double>& train_out,
		double learning_rate = 1e-3);

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

	size_t get_ninput (void) const;

	size_t get_noutput (void) const;

protected:
	db_net (const db_net& other, std::string& scope);

	db_net (db_net&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	void copy_helper (const db_net& other);

	void move_helper (db_net&& other);

	void clean_up (void);

	size_t n_input_;

	size_t n_output_;

	std::vector<rbm*> layers_;

	fc_layer* log_layer_ = nullptr;
};

}

#endif /* ROCNNET_DB_NET_HPP */
