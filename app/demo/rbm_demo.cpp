//
// Created by Mingkai Chen on 2017-07-19.
//

#include "compounds/rbm.hpp"
#include "mnist_data.hpp"

#ifdef __GNUC__
#include <unistd.h>
#endif

struct test_params
{
	size_t n_cont_div_ = 1;
	size_t n_epoch_ = 10;
	size_t n_hidden_ = 50;
	size_t n_batch_ = 20;
	size_t n_test_chain_ = 20;
	double learning_rate_ = 0.1;
	std::string outdir_ = ".";
	bool train_ = true;
};

void fit (rocnnet::rbm& model, std::vector<double> data, test_params params)
{
	size_t n_input = model.get_ninput();
	assert(0 == data.size() % n_input);
	size_t n_data = data.size() / n_input;
	size_t n_training_batches = n_data / params.n_batch_;

	nnet::placeholder<double> in(std::vector<size_t>{n_input, params.n_batch_}, "rbm_train_in");
	rocnnet::generators_t gens;
	nnet::updates_t trainers = model.train(gens, &in, params.learning_rate_ , params.n_cont_div_);
	trainers.push_back([gens](bool)
	{
		for (nnet::generator<double>* gen : gens)
		{
			gen->update({}); // re-randomize
		}
	});
	nnet::varptr<double> cost = model.get_pseudo_likelihood_cost(in);
	auto it = data.begin();
	double inbatch = params.n_batch_ * n_input;

	for (size_t i = 0; i < params.n_epoch_; i++)
	{
		double mean_cost = 0;
		for (size_t j = 0; j < n_training_batches; j++)
		{
			std::vector<double> batch(it + j * inbatch, it + (j + 1) * inbatch);
			in = batch;
			for (nnet::variable_updater<double>& trainer : trainers)
			{
				trainer(true);
			}

			mean_cost += nnet::expose<double>(cost)[0] / n_training_batches;
		}
		std::cout << "Training epoch " << i << ", cost is " << mean_cost << std::endl;
	}
}

void mnist_test (xy_data* train, xy_data* test, test_params params)
{
	std::string serialname = "rbm_test.pbx";
	std::string serialpath = params.outdir_ + "/" + serialname;

	std::vector<double> training_data(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> test_data(test->data_x_.begin(), test->data_x_.end());

	size_t n_input = train->shape_.first;

	rocnnet::rbm model(n_input, params.n_hidden_, "mnist_learner");

	if (params.train_)
	{
		model.initialize();
		fit(model, training_data, params);
	}
	else
	{
		model.initialize(serialpath, "rbm_demo");
	}

	const size_t plot_every = 1000;
	size_t n_test_input = test->shape_.first;
	size_t n_test_sample = test->shape_.second;
	std::uniform_int_distribution<int> dist(0, n_test_sample - params.n_test_chain_);
	size_t idx = dist(nnutils::get_generator());
	auto testbegin = test->data_x_.begin() + idx * n_test_input;
	std::vector<double> test_sample(testbegin, testbegin + params.n_test_chain_  * n_test_input);

	model.save(serialpath, "rbm_demo");

	nnet::placeholder<double> test_in(std::vector<size_t>{n_test_input, params.n_test_chain_});
	nnet::varptr<double> test_outsample = nnet::round(model(&test_in));
	nnet::varptr<double> test_generated_in = model.back(test_outsample); // < what we're interested in
	test_in = test_sample;

	std::vector<double> generation = nnet::expose<double>(test_generated_in);
	nnet::tensorshape generated_shape = test_generated_in->get_shape();

	for (double e : generation)
	{
		std::cout << e << ", ";
	}
	std::cout << std::endl;
	nnet::print_shape(generated_shape);
	std::cout << std::endl;
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
	nnet::varptr<double> test_out = model(&test_in);

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
	while ((c = getopt (argc, argv, "s:o:e:m:t:")) != -1)
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

	return 0;
}
