//
// Created by Mingkai Chen on 2017-04-20.
//

#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>

#include "gd_net.hpp"
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
	std::string serialname = "demotest.pbx";

	std::clock_t start;
	double duration;
	size_t n_train = 600;
	size_t n_test = 500;
	size_t n_in = 10;
	size_t n_out = 5;
	size_t n_batch = 1;
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(9, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_out, nnet::sigmoid<double>)
	};
	rocnnet::gd_net gdn(n_in, hiddens);
	rocnnet::gd_net* gdn2 = gdn.clone();
	rocnnet::gd_net gdn3(n_in, hiddens);
	gdn.initialize();
	gdn2->initialize();
	gdn3.initialize(serialname);
	gdn.learning_rate_ = gdn2->learning_rate_ =
	gdn3.learning_rate_ = 0.9;

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		std::cout << "training " << i << std::endl;
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		gdn.train(batch, batch_out);
	}
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << "training time: " << duration << " seconds" << std::endl;

	nnet::placeholder<double> in((std::vector<size_t>{n_in, 1}), "test_layerin");
	nnet::placeholder<double> in2((std::vector<size_t>{n_in, 1}), "test_layerin2");
	nnet::placeholder<double> in3((std::vector<size_t>{n_in, 1}), "test_layerin3");
	nnet::varptr<double> out = gdn(&in);
	nnet::varptr<double> out2 = (*gdn2)(&in2);
	nnet::varptr<double> out3 = gdn3(&in3);

	double good_err = 0;
	double bad_err = 0;
	double pretrained_err = 0;
	for (size_t i = 0; i < n_test; i++)
	{
		std::cout << "testing " << i << "\n";
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		in = batch;
		in2 = batch;
		in3 = batch;
		std::vector<double> res = nnet::expose<double>(out);
		std::vector<double> res2 = nnet::expose<double>(out2);
		std::vector<double> res3 = nnet::expose<double>(out3);
		double avgerr = 0;
		double avgerr2 = 0;
		double avgerr3 = 0;
		for (size_t i = 0, n = batch_out.size(); i < n; i++)
		{
			avgerr += std::abs(res[i] - batch_out[i]);
			avgerr2 += std::abs(res2[i] - batch_out[i]);
			avgerr3 += std::abs(res3[i] - batch_out[i]);
		}
		good_err += avgerr / res.size();
		bad_err += avgerr2 / res2.size();
		pretrained_err += avgerr3 / res3.size();
	}
	good_err *= 100.0 / (double) n_test;
	bad_err *= 100.0 / (double) n_test;
	pretrained_err *= 100.0 / (double) n_test;
	std::cout << "trained mlp error rate: " << good_err << "%\n";
	std::cout << "untrained mlp error rate: " << bad_err << "%\n";
	std::cout << "pretrained mlp error rate: " << pretrained_err << "%\n";

//	gdn.save(serialname);
	
	delete gdn2;

#ifdef EDGE_RCD
	rocnnet_record::erec::rec.to_csv<double>();
#endif /* EDGE_RCD */

	return 0;
}
