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

	std::random_device rnd_device;
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
	size_t n_train = 10000;
	size_t n_test = 100;
	size_t n_in = 10;
	size_t n_out = 5;
//	size_t n_batch = 5;
	size_t n_batch = 1;
	nnet::placeholder<double> in((std::vector<size_t>{n_in, n_batch}), "layerin");
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(9, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_out, nnet::sigmoid<double>)
	};
	rocnnet::ml_perceptron mlp(n_in, hiddens);
	mlp.initialize();

	nnet::varptr<double> output = mlp(&in);
	nnet::placeholder<double> expected_out(std::vector<size_t>{n_out, n_batch}, "expected_out");
	nnet::varptr<double> diff = output - nnet::varptr<double>(&expected_out);
	nnet::varptr<double> error = diff * diff;

	// training using gradient descent
	{
		nnet::inode<double>::GRAD_CACHE leafset;
		error->get_leaves(leafset);
		for (auto lit : leafset)
		{
			const nnet::tensor<double>* gres = error->get_gradient(lit.first);
		}
	}

	// train mlp to output input
//	start = std::clock();
//	for (size_t i = 0; i < n_train; i++)
//	{
//		std::cout << "training " << i << std::endl;
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		in = batch;
		expected_out = batch_out;
//	}
//	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//	std::cout<< "training time: " << duration << " seconds\n";

//	double err = 0;
//	for (size_t i = 0; i < test_size; i++)
//	{
//		std::cout << "testing " << i << "\n";
//		std::vector<double> batch = batch_generate(n_in, n_batch);
//		std::vector<double> batch_out = avgevry2(batch);
//		in = batch;
//		expected_out = batch_out;
//		std::vector<double> res = nnet::expose(diff);
//		double avgerr = 0;
//		for (double r : res)
//		{
//			avgerr += std::abs(r);
//		}
//		err += avgerr / res.size();
//	}

#ifdef EDGE_RCD
	rocnnet_record::erec::rec.to_csv<double>();
#endif /* EDGE_RCD */

	return 0;
}
