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

// batch gradient descent
// prone to overfitting
void gd_net::train (std::vector<VECS> sample) {
	if (NULL == mlp) return;
	std::vector<layer_perceptron* > layers = mlp->get_vars();
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
