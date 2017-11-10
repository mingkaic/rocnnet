//
// Created by Mingkai Chen on 2017-07-19.
//

#ifdef __GNUC__
#include <unistd.h>
#endif

#include "models/rbm.hpp"
#include "mnist_data.hpp"
#include "edgeinfo/csv_record.hpp"

struct test_params
{
	size_t n_cont_div_ = 15;
	size_t n_epoch_ = 10;
	size_t n_hidden_ = 50;
	size_t n_batch_ = 20;
	size_t n_test_chain_ = 20;
	size_t n_samples_ = 10;
	double learning_rate_ = 0.1;
	std::string outdir_ = ".";
	bool train_ = true;
};

std::string serialname = "rbm_test.pbx";

void fit (rocnnet::rbm& model, std::vector<double> data, test_params params)
{
	std::string serialpath = params.outdir_ + "/" + serialname;

	size_t n_input = model.get_ninput();
	assert(0 == data.size() % n_input);
	size_t n_data = data.size() / n_input;
	size_t n_training_batches = n_data / params.n_batch_;

	nnet::const_init<double> zinit(0);
	nnet::variable<double> persistent(std::vector<size_t>{params.n_hidden_, params.n_batch_}, zinit, "persistent");
	persistent.initialize();

	nnet::placeholder<double> in(std::vector<size_t>{n_input, params.n_batch_}, "rbm_train_in");
	rocnnet::update_cost_t training_res = model.train(&in, &persistent, params.learning_rate_ , params.n_cont_div_);

	nnet::variable_updater<double> trainer = training_res.first;
	nnet::varptr<double> cost = training_res.second;
	auto it = data.begin();
	double inbatch = params.n_batch_ * n_input;

	for (size_t i = 0; i < params.n_epoch_; i++)
	{
		double mean_cost = 0;
		for (size_t j = 0; j < n_training_batches; j++)
		{
			std::vector<double> batch(it + j * inbatch, it + (j + 1) * inbatch);
			in = batch;
			trainer(true); // train
			std::cout << "completed batch " << j << std::endl;

			mean_cost += nnet::expose<double>(cost)[0] / n_training_batches;
		}
		std::cout << "Training epoch " << i << ", cost is " << mean_cost << std::endl;

		model.save(serialpath, "rbm_demo"); // save in case of problems
	}

#ifdef CSV_RCD
if (rocnnet_record::record_status::rec_good)
{
	static_cast<rocnnet_record::csv_record*>(rocnnet_record::record_status::rec.get())->to_csv<double>();
}
#endif /* CSV_RCD */
}

void mnist_test (xy_data* train, xy_data* test, test_params params)
{
	std::string serialpath = params.outdir_ + "/" + serialname;

	std::vector<double> training_data(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> test_data(test->data_x_.begin(), test->data_x_.end());

	size_t n_input = train->shape_.first;

	rocnnet::rbm model(n_input, params.n_hidden_, "mnist_learner");

	if (params.train_)
	{
		model.initialize();
		fit(model, training_data, params);

		model.save(serialpath, "rbm_demo");
	}
	else
	{
		model.initialize(serialpath, "rbm_demo");
	}

	const size_t plot_every = 1000;
	size_t n_test_input = test->shape_.first;
	size_t n_test_sample = test->shape_.second;
	std::uniform_int_distribution<int> dist(0, n_test_sample - params.n_test_chain_);

	nnet::placeholder<double> test_in(std::vector<size_t>{n_test_input, params.n_test_chain_});
	nnet::varptr<double> test_generated_in = model.reconstruct_visible(&test_in);
	nnet::varptr<double> test_generated_sample = nnet::binomial_sample(1.0, test_generated_in);

	std::vector<std::vector<double> > outputchains;
	for (size_t i = 0; i < params.n_samples_; i++)
	{
		std::cout << "... plotting sample " << i << std::endl;
		size_t idx = dist(nnutils::get_generator());
		auto testbegin = test->data_x_.begin() + idx * n_test_input;
		std::vector<double> test_sample(testbegin, testbegin + params.n_test_chain_  * n_test_input);
		for (size_t j = 0; j < plot_every; j++)
		{
			test_in = test_sample;
			test_sample = nnet::expose<double>(test_generated_in);
		}

		outputchains.push_back(test_sample);
	}

	mnist_imageout(outputchains, test_in.get_shape().as_list(), params.n_test_chain_, params.n_samples_);
}

void small_test (test_params params)
{
	// using test from http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
	// <Harry Potter, Avatar, LOTR, Gladiator, Titanic, Glitter>
	std::vector<double> training_data = {
		1,1,1,0,0,0, // Alice likes
		1,0,1,0,0,0, // Bob
		1,1,1,0,0,0, // Carol
		0,0,1,1,1,0, // David
		0,0,1,1,0,0, // Eric
		0,0,1,1,1,0 // Fred
	};
	std::vector<double> test_data = {0,0,0,1,1,0};

	size_t n_input = 6;
	params.n_batch_ = 1;

	rocnnet::rbm model(n_input, params.n_hidden_, "movie_learner");
	model.initialize();

	fit(model, training_data, params);

	nnet::placeholder<double> test_in(std::vector<size_t>{n_input});
	nnet::varptr<double> test_out = model.reconstruct_visible(&test_in);

	test_in = test_data;
	std::vector<double> out = nnet::expose<double>(test_out);
	for (double o : out)
	{
		std::cout << o << ", ";
	}
	std::cout << std::endl;
}

int main (int argc, char** argv)
{
	// todo: replace with boost flags
	test_params params;
	bool mnist = false;
	std::experimental::optional<size_t> seed;
#ifdef __GNUC__ // use this gnu parser, since boost is too big for free-tier platforms
	int c;
	while ((c = getopt (argc, argv, "s:o:e:m:k:t:")) != -1)
	{
		switch(c)
		{
			case 's':
				seed = atoi(optarg);
				break;
			case 'o': // output directory
				params.outdir_ = std::string(optarg);
				break;
			case 'e': // epoch training iteration
				params.n_epoch_ = atoi(optarg);
				break;
			case 'm':
				mnist = true;
				break;
			case 'k': // k-CD or k-PCD
				params.n_cont_div_ = atoi(optarg);
				break;
			case 't':
				params.train_ = false;
				break;
		}
	}
#else
	if (argc > 1)
	{
		params.outdir_ = std::string(argv[1]);
	}
	if (argc > 2)
	{
		params.n_epoch_ = atoi(argv[2]);
	}
#endif
	if (seed)
	{
		nnutils::seed_generator(*seed);
	}

	if (mnist)
	{
		try
		{
			Py_Initialize();
			np::initialize();
			std::vector<xy_data*> datasets = get_mnist_data();

			xy_data* training_set = datasets[0];
			xy_data* testing_set = datasets[2];

			mnist_test(training_set, testing_set, params);

			for (xy_data* dataset : datasets)
			{
				delete dataset;
			}
		}
		catch(const bp::error_already_set&)
		{
			std::cerr << ">>> Error! Uncaught exception:\n";
			PyErr_Print();
			return 1;
		}
	}
	else
	{
		small_test(params);
	}

	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}
