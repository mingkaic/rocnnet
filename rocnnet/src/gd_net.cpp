//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../include/gd_net.hpp"

#ifdef gd_net_hpp

namespace nnet
{

group<double>* ad_hoc_gd_setup (double learning_rate,
		ivariable<double>* train_in,
		varptr<double> diff_func,
		varptr<double> batch_size,
		std::vector<HID_PAIR>& layers,
		std::queue<varptr<double> >& layer_out,
		std::stack<varptr<double> >& prime_out)
{
	// err_n = (out-y)*act'(z_n)
	// where z_n is the input to the nth layer (last)
	// and act is the activation operation
	varptr<double> err = diff_func * prime_out.top();
	prime_out.pop();
	std::stack<varptr<double> > errs;
	errs.push(err);

	auto term = --layers.rend();
	for (auto rit = layers.rbegin();
		 term != rit; rit++)
	{
		HID_PAIR hp = *rit;
		// err_i = matmul(err_i+1, transpose(weight_i))*f'(z_i)
		ivariable<double>* weight_i = hp.first->get_variables().first;
		// weight is input by output, err is output by batch size, so we expect mres to be input by batch size
		varptr<double> mres = matmul<double>::build(err, weight_i, false ,true);
		err = mres * prime_out.top();
		prime_out.pop();
		errs.push(err);
	}

	varptr<double> learn_batch = learning_rate / batch_size;
	group<double>* updates = new group<double>();
	ivariable<double>* output = train_in;
	for (HID_PAIR hp : layers)
	{
		err = errs.top();
		errs.pop();

		// dweights = learning*matmul(transpose(layer_in), err)
		// dbias = learning*err
		varptr<double> cost = matmul<double>::build(output, err, true);
		varptr<double> dweights = cost * learn_batch;

		// expecting err to be output by batchsize, compress along batchsize
		varptr<double> compressed_err = new compress<double>(err, 1);
		varptr<double> dbias = compressed_err * learn_batch;

		// update weights and bias
		// weights -= dweights, bias -= dbias
		WB_PAIR wb = hp.first->get_variables();
		iexecutor<double>* w_evok = new assign_sub<double>(wb.first, dweights);
		iexecutor<double>* b_evok = new assign_sub<double>(wb.second, dbias);

		// store for update
		updates->add(w_evok);
		updates->add(b_evok);

		output = layer_out.front();
		layer_out.pop();
	}
	return updates;
}

void gd_net::train_set_up (void)
{
	// follow () operator for ml_perceptron except store hypothesis
	std::queue<varptr<double> > layer_out;
	std::stack<varptr<double> > prime_out;

	// not so generic setup
	varptr<double> output = train_in_;
	for (HID_PAIR hp : layers)
	{
		varptr<double> hypothesis = (*hp.first)(output);
		output = (hp.second)(hypothesis);
		layer_out.push(output);
		// act'(z_i)
		varptr<double> grad = new derive<double>(output, hypothesis);
		prime_out.push(grad);
	}

	// preliminary setup for any update algorithm
	diff_ = output - expected_out;

	// optimizer_ and updates are two separate executors (updates technically NOT an executor)
	// TODO: get rid of updates at some point (once optimizer proves to work)
	if (nullptr == optimizer_)
	{
		updates = ad_hoc_gd_setup(
			learning_rate,
			train_in_, diff,
			batch_size, this->layers,
			layer_out, prime_out);
	}
	else
	{
		optimizer_->set_root(diff * diff);
		optimizer_->freeze();
	}
}

gd_net::gd_net (const gd_net& net, std::string scope) : 
	ml_perceptron(net, scope)
{
	n_input = net.n_input;
	learning_rate = net.learning_rate;
	batch_size = net.batch_size->clone();
	train_in_ = net.train_in_->clone();
	expected_out = net.expected_out->clone();
	train_set_up();
}

gd_net::gd_net (size_t n_input,
	std::vector<IN_PAIR> hiddens,
	OPTIMIZER<double> optimizer,
	std::string scope) : 
	ml_perceptron(n_input, hiddens, scope), n_input(n_input), optimizer_(optimizer) 
{
	size_t n_out = hiddens.back().first;
	batch_size = new placeholder<double>(std::vector<size_t>{0}, "batch_size");
	train_in_ = new placeholder<double>(std::vector<size_t>{n_input, 0}, "train_in");
	expected_out = new placeholder<double>(std::vector<size_t>{n_out, 0}, "expected_out");
	train_set_up();
}

// batch gradient descent
// 1/m*sum_m(Err(X, Y)) once matrix operations support auto derivation
// then apply cost funct to grad desc alg:
// new weight = old weight - learning_rate * cost func gradient over old weight
// same thing with bias (should experience no rocnnet decrease due to short circuiting)
void gd_net::train (std::vector<double> train_in, std::vector<double> expected_out) 
{
	(*this->batch_size) = std::vector<double>{(double) train_in.size() / n_input};
	(*train_in_) = train_in;
	(*this->expected_out) = expected_out;
	// trigger update
	if (optimizer_)
	{
		optimizer_->execute();
	}
	else
	{
		updates->execute();
	}
	if (record_training)
	{
		std::vector<double> errs = nnet::expose<double>(diff_);
		double avg_err = 0;
		for (double e : errs) {
			avg_err += std::abs(e);
		}
		avg_err /= errs.size();
		std::cout << "err: " << avg_err << "\n";
	}
}

}

#endif /* gd_net_hpp */
