//
// Created by Mingkai Chen on 2017-04-20.
//

#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>

#include "mlp.hpp"
#include "edgeinfo/comm_record.hpp"

static std::vector<double> batch_generate (size_t n, size_t batchsize)
{
	size_t total = n * batchsize;

//	std::random_device rnd_device;
	std::default_random_engine rnd_device;
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(rnd_device());
	std::uniform_real_distribution<double> dist(0, 1);

	auto gen = std::bind(dist, mersenne_engine);
	std::vector<double> vec(total);
	std::generate(std::begin(vec), std::end(vec), gen);
	return vec;
}

static std::vector<double> avgevry2 (std::vector<double>& in)
{
	std::vector<double> out;
	for (size_t i = 0, n = in.size()/2; i < n; i++)
	{
		double val = (in.at(2*i) + in.at(2*i+1)) / 2;
		out.push_back(val);
	}
	return out;
}

int main (int argc, char** argv)
{
	std::clock_t start;
	double duration;
	size_t n_train = 600;
	size_t n_test = 500;
	size_t n_in = 10;
	size_t n_out = 5;
//	size_t n_batch = 5;
	size_t n_batch = 1;
	nnet::placeholder<double> in((std::vector<size_t>{n_in, n_batch}), "layerin");
	nnet::placeholder<double> in2((std::vector<size_t>{n_in, n_batch}), "layerin2");
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(9, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_out, nnet::sigmoid<double>)
	};
	rocnnet::ml_perceptron mlp(n_in, hiddens);
	rocnnet::ml_perceptron* mlp2 = mlp.clone();
	mlp.initialize();
	mlp2->initialize();

	nnet::varptr<double> output = mlp(&in);
	nnet::placeholder<double> expected_out(std::vector<size_t>{n_out, n_batch}, "expected_out");
	nnet::varptr<double> diff = nnet::varptr<double>(&expected_out) - output;
	nnet::varptr<double> error = diff * diff;
	nnet::iconnector<double>* topdog = static_cast<nnet::iconnector<double>*>(error.get());

	nnet::varptr<double> out2 = (*mlp2)(&in2);
	nnet::placeholder<double> expected_out2(std::vector<size_t>{n_out, n_batch}, "expected_out2");
	nnet::varptr<double> diff2 = nnet::varptr<double>(&expected_out2) - out2;

	double learning_rate = 0.9;
	// training using gradient descent
	std::vector<nnet::variable_updater<double> > update;
	nnet::inode<double>::GRAD_CACHE leafset;
	error->get_leaves(leafset);
	for (auto lit : leafset)
	{
		nnet::variable<double>* Wb = lit.first;
		nnet::varptr<double> gres = error->get_gradient(Wb);
		// Wb = Wb + learning_rate * gres
		update.push_back(Wb->assign_add(gres * learning_rate));
	}

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		std::cout << "training " << i << std::endl;
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		in = batch;
		expected_out = batch_out;
		topdog->update_status(true); // freeze
		for (auto& trainer : update)
		{
			trainer();
		}
		topdog->update_status(false); // update again
	}
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << "training time: " << duration << " seconds" << std::endl;

	double good_err = 0;
	double bad_err = 0;
	for (size_t i = 0; i < n_test; i++)
	{
		std::cout << "testing " << i << "\n";
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		in = batch;
		in2 = batch;
		expected_out = batch_out;
		expected_out2 = batch_out;
		std::vector<double> res = nnet::expose<double>(diff);
		std::vector<double> res2 = nnet::expose<double>(diff2);
		double avgerr = 0;
		for (double r : res)
		{
			avgerr += std::abs(r);
		}
		double avgerr2 = 0;
		for (double r : res2)
		{
			avgerr2 += std::abs(r);
		}
		good_err += avgerr / res.size();
		bad_err += avgerr2 / res2.size();
	}
	good_err *= 100.0 / (double) n_test;
	bad_err *= 100.0 / (double) n_test;
	std::cout << "trained mlp error rate: " << good_err << "%\n";
	std::cout << "untrained mlp error rate: " << bad_err << "%\n";
	
	delete mlp2;

#ifdef EDGE_RCD
	rocnnet_record::erec::rec.to_csv<double>();
#endif /* EDGE_RCD */

	return 0;
}
