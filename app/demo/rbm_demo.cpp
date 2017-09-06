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

void test_rbm(xy_data* train, xy_data* test, test_params params)
{
	std::vector<double> training_data(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> test_data(test->data_x_.begin(), test->data_x_.end());

	size_t n_input = train->shape_.first;
	size_t n_hidden = 50;

	double learning_rate = 0.1;

	rocnnet::rbm model(n_input, n_hidden, "mnist_learner");
	model.initialize();

	rocnnet::fit(model, training_data, {params.n_epoch_, 1, learning_rate});

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

int main (int argc, char** argv)
{
	try
	{
		Py_Initialize();
		np::initialize();

		std::vector<xy_data*> datasets = get_mnist_data();

		xy_data* training_set = datasets[0];
		xy_data* testing_set = datasets[2];

		// todo: replace with boost flags
		test_params params;
#ifdef __GNUC__ // use this gnu parser, since boost is too big for free-tier platforms
		int c;
		while ((c = getopt (argc, argv, "o:e:")) != -1)
		{
			switch(c)
			{
				case 'o': // output directory
					params.outdir_ = std::string(optarg);
					break;
				case 'e': // epoch training iteration
					params.n_epoch_ = atoi(optarg);
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

		test_rbm(training_set, testing_set, params);

		for (xy_data* dataset : datasets)
		{
			delete dataset;
		}

		return 0;
	}
	catch(const bp::error_already_set&)
	{
		std::cerr << ">>> Error! Uncaught exception:\n";
		PyErr_Print();
		return 1;
	}
}
