//
// Created by Mingkai Chen on 2017-07-26.
//

#include "compounds/rbm.hpp"

#ifndef ROCNNET_DBN_HPP
#define ROCNNET_DBN_HPP

namespace rocnnet
{

struct dbn_param
{
	size_t n_epoch_ = 10;
	size_t n_cont_div_ = 1;
	double learning_rate_ = 1e-3;
};

class dbn : public icompound
{
public:
	dbn (size_t n_input, std::vector<size_t> hiddens,
		dbn_param train_param,
		std::string scope = "DBN");

	virtual ~dbn (void);

	dbn* clone (std::string scope = "") const;

	dbn* move (void);

	dbn& operator = (const dbn& other);

	dbn& operator = (dbn&& other);

	// PLACEHOLDER CONNECTION
	// input are expected to have shape n_input by batch_size
	// outputs are expected to have shape output by batch_size
	nnet::varptr<double> operator () (nnet::inode<double>* input);

	virtual std::vector<nnet::variable<double>*> get_variables (void) const;

	size_t get_ninput (void) const;

	size_t get_noutput (void) const;

protected:
	dbn (const dbn& other, std::string& scope);

	dbn (dbn&& other);

	virtual ilayer* clone_impl (std::string& scope) const;

	virtual ilayer* move_impl (void);

private:
	void copy_helper (const dbn& other);

	void move_helper (dbn&& other);

	void clean_up (void);

	size_t n_input_;

	size_t n_output_;

	std::vector<rbm*> layers_;
};

}

#endif /* ROCNNET_DBN_HPP */
