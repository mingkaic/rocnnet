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
	size_t n_epoch_ = 10;
	std::string outdir_ = ".";
};

void fit (rocnnet::rbm& model, std::vector<double> data, test_params params, size_t n_batch, double lr, size_t n_cont_div)
{
	size_t n_input = model.get_ninput();
	assert(0 == data.size() % n_input);
	size_t n_data = data.size() / n_input;
	size_t n_training_batches = n_data / n_batch;

	nnet::placeholder<double> in(std::vector<size_t>{n_input, n_batch}, "rbm_train_in");
	rocnnet::generators_t gens;
	nnet::updates_t trainers = model.train(gens, &in, lr, n_cont_div);
	trainers.push_back([gens](bool)
	{
		for (nnet::generator<double>* gen : gens)
		{
			gen->update({}); // re-randomize
		}
	});
	nnet::varptr<double> cost = model.get_pseudo_likelihood_cost(in);
	auto it = data.begin();
	double inbatch = n_batch * n_input;

	for (size_t i = 0; i < params.n_epoch_; i++)
	{
		size_t mean_cost = 0;
		for (size_t j = 0; j < n_training_batches; j++)
		{
			std::vector<double> batch(it + j * inbatch, it + (j + 1) * inbatch);
			in = batch;
			for (nnet::variable_updater<double>& trainer : trainers)
			{
				trainer(true);
			}

			mean_cost += nnet::expose<double>(cost)[0];
		}
		std::cout << "Training epoch " << i << ", cost is " << mean_cost / n_training_batches << std::endl;
	}
}

void mnist_test (xy_data* train, xy_data* test, test_params params)
{
	std::vector<double> training_data(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> test_data(test->data_x_.begin(), test->data_x_.end());

	size_t n_input = train->shape_.first;
	size_t n_hidden = 50;

	double learning_rate = 0.1;

	rocnnet::rbm model(n_input, n_hidden, "mnist_learner");
	model.initialize();

	fit(model, training_data, params, 20, learning_rate, 1);

	nnet::placeholder<double> test_in(std::vector<size_t>{n_input});
	nnet::varptr<double> test_out = model(&test_in);

	size_t n_test = test_data.size() / n_input;
	for (size_t i = 0; i < n_test; i++)
	{
		auto it = test_data.begin() + i * n_input;
		auto et = test_data.begin() + (i + 1) * n_input;
		std::vector<double> itest(it, et);

		test_in = itest; // push update
		std::vector<double> out = nnet::expose<double>(test_out);
		for (double o : out)
		{
			std::cout << o << ", ";
		}
		std::cout << std::endl;
	}
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
	size_t n_hidden = 2;

	double learning_rate = 0.1;

	rocnnet::rbm model(n_input, n_hidden, "movie_learner");
	model.initialize();

	fit(model, training_data, params, 20, learning_rate, 1);

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
#ifdef __GNUC__ // use this gnu parser, since boost is too big for free-tier platforms
	int c;
	while ((c = getopt (argc, argv, "o:e:m:")) != -1)
	{
		switch(c)
		{
			case 'o': // output directory
				params.outdir_ = std::string(optarg);
				break;
			case 'e': // epoch training iteration
				params.n_epoch_ = atoi(optarg);
				break;
			case 'm':
				mnist = true;
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
