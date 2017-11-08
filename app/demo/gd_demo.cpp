//
// Created by Mingkai Chen on 2017-04-20.
//

#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>

#ifdef __GNUC__
#include <unistd.h>
#endif

#include "models/gd_net.hpp"
#include "edgeinfo/csv_record.hpp"

static std::default_random_engine rnd_device(std::time(NULL));

static std::vector<double> batch_generate (size_t n, size_t batchsize)
{
	size_t total = n * batchsize;

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
	std::string outdir = ".";
	size_t n_train = 3000;
	size_t n_test = 500;
	size_t seed_val;
	bool seed = false;
	bool save = false;

#ifdef __GNUC__ // use this gnu parser, since boost is too big for free-tier platforms
	int c;
	while ((c = getopt (argc, argv, "o:r:t:s:w:")) != -1)
	{
		switch(c)
		{
			case 'o': // output directory
				outdir = std::string(optarg);
				break;
			case 'r': // training iterations
				n_train = atoi(optarg);
				break;
			case 't': // testing iterations
				n_test = atoi(optarg);
				break;
			case 's': // seed value
				seed_val = atoi(optarg);
				seed = true;
				break;
			case 'w': // save test file
				save = true;
				break;
		}
	}
#else
	if (argc > 1)
	{
		outdir = std::string(argv[1]);
	}
	if (argc > 2)
	{
		n_train = atoi(argv[2]);
	}
	if (argc > 3)
	{
		n_test = atoi(argv[3]);
	}
	if (argc > 4)
	{
		seed_val = atoi(argv[4]);
		seed = true;
	}
#endif

	if (seed)
	{
		rnd_device.seed(seed_val);
		nnutils::seed_generator(seed_val);
	}

	std::string serialname = "gd_test.pbx";
	std::string serialpath = outdir + "/" + serialname;

	size_t n_in = 10;
	size_t n_out = n_in / 2;
	size_t n_batch = 3;
	size_t show_every_n = 500;
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once models copy is established
		rocnnet::IN_PAIR(9, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_out, nnet::sigmoid<double>)
	};
	nnet::vgb_updater bgd(0.9); // learning rate = 0.9
	rocnnet::mlp* brain = new rocnnet::mlp(n_in, hiddens);
	rocnnet::gd_net untrained_gdn(brain, bgd, "untrained_gd_net");
	untrained_gdn.initialize();
	rocnnet::gd_net trained_gdn(untrained_gdn, "trained_gd_net");

	// use this to satisfy moving test coverage for gd net
	rocnnet::gd_net temporary_gdn(untrained_gdn, "temporary_gd_net");
	temporary_gdn.initialize(serialpath, "gd_demo");
	rocnnet::gd_net pretrained_gdn(std::move(temporary_gdn), "pretrained_gd_net");

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		if (i % show_every_n == show_every_n-1) std::cout << "training " << i+1 << std::endl;
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		trained_gdn.train(batch, batch_out);
	}
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << "training time: " << duration << " seconds" << std::endl;

	int exit_status = 0;
	// exit code:
	//	0 = fine
	//	1 = training error rate is wrong
	double untrained_err = 0;
	double trained_err = 0;
	double pretrained_err = 0;
	for (size_t i = 0; i < n_test; i++)
	{
		if (i % show_every_n == show_every_n-1) std::cout << "testing " << i+1 << "\n";
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		std::vector<double> untrained_res = untrained_gdn(batch);
		std::vector<double> trained_res = trained_gdn(batch);
		std::vector<double> pretrained_res = pretrained_gdn(batch);
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
	untrained_err /= (double) n_test;
	trained_err /= (double) n_test;
	pretrained_err /= (double) n_test;
	std::cout << "untrained mlp error rate: " << untrained_err * 100 << "%\n";
	std::cout << "trained mlp error rate: " << trained_err * 100 << "%\n";
	std::cout << "pretrained mlp error rate: " << pretrained_err * 100 << "%\n";

	// fails if cumulative training is over threshold=250, 
	// and trained is inferior to untrained
	if (n_train > 250 && untrained_err < trained_err)
	{
		exit_status = 1;
	}

	if (exit_status == 0 && save)
	{
		trained_gdn.save(serialpath, "gd_demo");
	}

#ifdef CSV_RCD
if (rocnnet_record::record_status::rec_good)
	static_cast<rocnnet_record::csv_record*>(rocnnet_record::record_status::rec.get())->to_csv<double>(trained_gdn.get_error());
#endif /* CSV_RCD */

	google::protobuf::ShutdownProtobufLibrary();

	return exit_status;
}
