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
	std::string outdir = ".";
	if (argc > 1)
	{
		outdir = std::string(argv[1]);
	}
	std::string serialname = "demotest.pbx";
	std::string serialpath = outdir + "/" + serialname;

	std::clock_t start;
	double duration;
	size_t n_train = 600;
	size_t n_test = 500;
	size_t n_in = 10;
	size_t n_out = 5;
	size_t n_batch = 3;
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(9, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_out, nnet::sigmoid<double>)
	};
	nnet::vgb_updater bgd;
	bgd.learning_rate_ = 0.9;
	rocnnet::gd_net untrained_gdn(n_in, hiddens, bgd);
	rocnnet::gd_net* trained_gdn = untrained_gdn.clone();
	rocnnet::gd_net pretrained_gdn(n_in, hiddens, bgd);
	untrained_gdn.initialize();
	trained_gdn->initialize();
	pretrained_gdn.initialize(serialpath);

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		if (i % 10 == 9) std::cout << "training " << i+1 << std::endl;
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		trained_gdn->train(batch, batch_out);
	}
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << "training time: " << duration << " seconds" << std::endl;

	nnet::placeholder<double> untrained_in((std::vector<size_t>{n_in, n_batch}), "test_untrain_layerin");
	nnet::placeholder<double> trained_in((std::vector<size_t>{n_in, n_batch}), "test_train_layerin");
	nnet::placeholder<double> pretrained_in((std::vector<size_t>{n_in, n_batch}), "test_pretrain_layerin");
	nnet::varptr<double> untrained_out = untrained_gdn(&untrained_in);
	nnet::varptr<double> trained_out = (*trained_gdn)(&trained_in);
	nnet::varptr<double> pretrained_out = pretrained_gdn(&pretrained_in);

	double untrained_err = 0;
	double trained_err = 0;
	double pretrained_err = 0;
	for (size_t i = 0; i < n_test; i++)
	{
		if (i % 10 == 9) std::cout << "testing " << i+1 << "\n";
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		untrained_in = batch;
		trained_in = batch;
		pretrained_in = batch;
		std::vector<double> untrained_res = nnet::expose<double>(untrained_out);
		std::vector<double> trained_res = nnet::expose<double>(trained_out);
		std::vector<double> pretrained_res = nnet::expose<double>(pretrained_out);
		double untrained_avgerr = 0;
		double trained_avgerr = 0;
		double pretrained_avgerr = 0;
		for (size_t i = 0, n = batch_out.size(); i < n; i++)
		{
			untrained_avgerr += std::abs(untrained_res[i] - batch_out[i]);
			trained_avgerr += std::abs(trained_res[i] - batch_out[i]);
			pretrained_avgerr += std::abs(pretrained_res[i] - batch_out[i]);
		}
		untrained_err += untrained_avgerr / untrained_res.size();
		trained_err += trained_avgerr / trained_res.size();
		pretrained_err += pretrained_avgerr / pretrained_res.size();
	}
	untrained_err *= 100.0 / (double) n_test;
	trained_err *= 100.0 / (double) n_test;
	pretrained_err *= 100.0 / (double) n_test;
	std::cout << "untrained mlp error rate: " << untrained_err << "%\n";
	std::cout << "trained mlp error rate: " << trained_err << "%\n";
	std::cout << "pretrained mlp error rate: " << pretrained_err << "%\n";

	trained_gdn->save(serialpath);
	
	delete trained_gdn;

#ifdef EDGE_RCD
	rocnnet_record::erec::rec.to_csv<double>();
#endif /* EDGE_RCD */

	return 0;
}
