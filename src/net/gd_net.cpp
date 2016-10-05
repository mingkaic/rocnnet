//
//  gd_net.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/nnet.hpp"

#ifdef gd_net_hpp

namespace nnet {

void gd_net::clear_ownership (void) {
	for (ivariable<double>* mine : ownership) {
		delete mine;
	}
	ownership.clear();
}

gd_net::gd_net (size_t n_input,
	std::vector<IN_PAIR> hiddens,
	std::string scope)
	: ml_perceptron(n_input, hiddens, scope) {}

gd_net::~gd_net (void) {
	clear_ownership();
}

// batch gradient descent
// TODO upgrade to using ioperation's derive method for cost function
// 1/m*sum_m(Err(X, Y)) once matrix operations support auto derivation
// then apply cost funct to grad desc alg:
// new weight = old weight - learning_rate * cost func derive over old weight
// same thing with bias (should experience no performance decrease due to short circuiting)
void gd_net::train (ivariable<double>& expected_out) {
	// follow () operator for ml_perceptron except store hypothesis
	std::stack<ivariable<double>*> act_prime;
	ivariable<double>* output = this->in_place;
	for (size_t i = 0; i < hypothesi.size(); i++) {
		// act'(z_i)
		ivariable<double>* prime = new derive<double>(*this->layers[i].second, *hypothesi[i]);
		ownership.insert(prime);
		act_prime.push(prime);
		output = this->layers[i].second;
	}
	// err_n = (out-y)*act'(z_n)
	// where z_n is the input to the nth layer (last)
	// and act is the activation operation
	// take ownership
	ivariable<double>* diff = new sub<double>(*output, expected_out);
	ownership.insert(diff);
	ivariable<double>* err = new mul<double>(*diff, *act_prime.top());
	ownership.insert(err);
	act_prime.pop();

	std::stack<ivariable<double>*> errs;
	errs.push(err);
	auto term = --this->layers.rend();
	for (auto rit = this->layers.rbegin();
		 term != rit; rit++) {
		HID_PAIR hp = *rit;
		// err_i = matmul(err_i+1, transpose(weight_i))*f'(z_i)
		ivariable<double>* weight_i = hp.first->get_variables().first;
        // weight is input by output, err is output by batch size, so we expect mres to be input by batch size
		ivariable<double>* mres = new matmul<double>(*err, *weight_i, false ,true);
		ownership.insert(mres);
		err = new mul<double>(*mres, *act_prime.top());
		ownership.insert(err);
		act_prime.pop();
		errs.push(err);
	}


	double learn_batch = learning_rate / expected_out.get_shape().as_list()[1];
	output = this->in_place;
	for (HID_PAIR hp : this->layers) {
		err = errs.top();
		errs.pop();
		WB_PAIR wb = hp.first->get_variables();

		// dweights = learning*matmul(transpose(layer_in), err)
		// dbias = learning*err
		// expecting err to be output by batchsize, compress along batchsize
		ivariable<double>* compressed_err = new compress<double>(*err, 1);
		ownership.insert(compressed_err);

		ivariable<double>* cost = new matmul<double>(*output, *compressed_err, true);
		ownership.insert(cost);
		ivariable<double>* dweights = new mul<double>(*cost, learn_batch);
		ownership.insert(dweights);
		ivariable<double>* dbias = new mul<double>(*compressed_err, learn_batch);
		ownership.insert(dbias);

		// update weights and bias
		// weights -= avg_row(dweights), bias -= avg_row(dbias)
		const tensor<double>& dwt = dweights->eval();
		const tensor<double>& dbt = dbias->eval();
		wb.first->update(dwt, shared_cnnet::op_sub<double>);
		wb.second->update(dbt, shared_cnnet::op_sub<double>);

		output = hp.second;
	}

	clear_ownership(); // all intermediate vars are used once anyways
}

// batch gradient descent
// prone to overfitting
void gd_net::train (std::vector<VECS> sample) {
	std::vector<layer_perceptron* > layers = this->get_vars();
	std::vector<std::pair<V_MATRIX, std::vector<double> > > storage;
	for (layer_perceptron* lp : layers) {
		size_t n_input = lp->get_n_input();
		size_t n_output = lp->get_n_output();
		V_MATRIX vm;
		vm.insert(vm.begin(), n_input, std::vector<double>(n_output));
		storage.push_back(std::pair<V_MATRIX, std::vector<double> >(vm, std::vector<double>(n_output)));
	}
	double A = learning_rate/sample.size();
	for (VECS io_pair : sample) {
		std::vector<double> exout = io_pair.second;
		std::vector<double> output = io_pair.first;
		std::stack<std::vector<double> > dzs; // {f'(z)}
		std::vector<std::vector<double> > as;
		for (layer_perceptron* lp : layers) {
			as.push_back(output);
			std::vector<double> z = lp->hypothesis(output);
			for (size_t i = 0; i < z.size(); i++) {
				z[i] = lp->op.grad(z[i]);
			}
			dzs.push(z);
			output = (*lp)(output);
		}
		// err_n = (a_n-y)*f'(z_n)
		std::vector<double> err = dzs.top();
		dzs.pop();
		for (size_t i = 0; i < err.size(); i++) {
			err[i] *= (output[i]-exout[i]);
		}
		std::stack<std::vector<double> > errs;
		errs.push(err);

		for (auto rit = ++layers.rbegin();
			layers.rend() != rit; rit++) {
			std::vector<double> dz = dzs.top();
			dzs.pop();

			// err_i = matmul(transpose(weight_i), err_i+1)*f'(z_i)
			std::pair<V_MATRIX&, double*> Wb_pair = (*rit)->get_vars();
			std::vector<double> diff;
			for (size_t x = 0; x < dz.size(); x++) {
				diff.push_back(0);
				for (size_t y = 0; y < err.size(); y++) {
					diff[x] += Wb_pair.first[x][y] * err[y];
				}
				diff[x] *= dz[x];
			}
			err = diff;
			errs.push(err);
		}

		for (size_t i = 0; i < layers.size(); i++) {
			std::vector<double> a = as[i];
			err = errs.top();
			errs.pop();

			// dweights+=learning*matmul(err, np.transpose(a))/#sample
			// dbias+=learning*err/#sample
			for (size_t y = 0; y < err.size(); y++) {
				for (size_t x = 0; x < a.size(); x++) {
					storage[i].first[x][y] += A*err[y]*a[x];
				}
				storage[i].second[y] += A*err[y];
			}
		}
	}

	// weights-=dweights, bias-=dbias
	for (size_t i = 0; i < layers.size(); i++) {
		size_t n_input = layers[i]->get_n_input();
		size_t n_output = layers[i]->get_n_output();
		std::pair<V_MATRIX&, double*> Wb_pair = layers[i]->get_vars();
		for (size_t y = 0; y < n_output; y++) {
			for (size_t x = 0; x < n_input; x++) {
				Wb_pair.first[x][y] -= storage[i].first[x][y];
			}
			Wb_pair.second[y] -= storage[i].second[y];
		}
	}
}

}

#endif /* gd_net_hpp */
