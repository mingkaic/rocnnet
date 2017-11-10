//
// Created by Mingkai Chen on 2017-09-21.
//

#include <random>

#ifdef __GNUC__
#include <unistd.h>
#endif

#include "models/db_net.hpp"
#include "mnist_data.hpp"
#include "edgeinfo/csv_record.hpp"

static std::default_random_engine rnd_device(std::time(NULL));

struct test_params
{
	size_t pretrain_epochs_ = 100;
	double pretrain_lr_ = 0.01;
	size_t training_epochs_ = 1000;
	double training_lr_ = 0.1;
	size_t n_cont_div_ = 15;
	size_t n_batch_ = 20;
	std::vector<size_t> hiddens_ = { 1000, 1000, 1000 };
	bool pretrain_ = true;
	bool train_ = true;
	std::string outdir_ = ".";
};

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

std::string serialname = "dbn_test.pbx";

void pretrain (rocnnet::db_net& model, size_t n_input,
	std::vector<double> data, test_params params, std::string test_name)
{
	std::string serialpath = params.outdir_ + "/" + serialname;
	size_t n_data = data.size() / n_input;
	size_t n_train_batches = n_data / params.n_batch_;
	nnet::placeholder<double> pretrain_in(std::vector<size_t>{n_input, params.n_batch_}, "pretrain_in");
	double inbatch = params.n_batch_ * n_input;

	std::cout << "... getting the pretraining functions" << std::endl;
	rocnnet::pretrain_t pretrainers = model.pretraining_functions(
		pretrain_in, params.pretrain_lr_ , params.n_cont_div_);

	std::cout << "... pre-training the model" << std::endl;
	auto it = data.begin();
	for (size_t pidx = 0; pidx < pretrainers.size(); pidx++)
	{
		auto ptit = pretrainers[pidx];
		nnet::variable_updater<double> trainer = ptit.first;
		nnet::varptr<double> cost = ptit.second;
		for (size_t e = 0; e < params.pretrain_epochs_; e++)
		{
			double mean_cost = 0;
			for (size_t i = 0; i < n_train_batches; i++)
			{
				std::vector<double> batch(it + i * inbatch, it + (i + 1) * inbatch);
				pretrain_in = batch;
				trainer(true);
				std::cout << "layer " << pidx << " epoch " << e << " completed batch " << i << std::endl;

				mean_cost += nnet::expose<double>(cost)[0] / n_train_batches;
			}
			std::cout << "pre-trained layer " << pidx << " epoch "
				<< e << " has mean cost " << mean_cost << std::endl;

			model.save(serialpath, "dbn_" + test_name + "_pretrain"); // save in case of problems
		}
	}

#ifdef CSV_RCD
if (rocnnet_record::record_status::rec_good)
{
	static_cast<rocnnet_record::csv_record*>(rocnnet_record::record_status::rec.get())->to_csv<double>();
}
#endif /* CSV_RCD */
}

void finetune (rocnnet::db_net& model, xy_data* train,
	xy_data* valid, xy_data* test, size_t n_input,
	size_t n_output, test_params params)
{
	size_t n_train_batches = train->shape_.first;

	std::vector<double> training_data_x(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> training_data_y(train->data_y_.begin(), train->data_y_.end());

	std::cout << "... getting the finetuning functions" << std::endl;
	nnet::placeholder<double> finetune_in(std::vector<size_t>{n_input, params.n_batch_}, "finetune_in");
	nnet::placeholder<double> finetune_out(std::vector<size_t>{n_output, params.n_batch_}, "finetune_out");
	rocnnet::update_cost_t tuner = model.build_finetune_functions(finetune_in, finetune_out, params.training_lr_);
	nnet::variable_updater<double> train_update = tuner.first;
	nnet::varptr<double> train_cost = tuner.second;
	nnet::varptr<double> train_loss = nnet::reduce_mean(train_cost);

	std::cout << "... finetuning the model" << std::endl;
	size_t patience = 4 * n_train_batches;
	size_t patience_increase = 2;
	size_t validation_frequency = std::min(n_train_batches, patience / 2);
	double improvement_threshold = 0.995;
	double best_validation_loss = std::numeric_limits<double>::infinity();
	double test_score = 0;
	bool keep_looping = true;
	size_t best_iter = 0;

	size_t inbatch = params.n_batch_ * n_input;
	size_t outbatch = params.n_batch_ * n_output;
	auto xit = training_data_x.begin();
	auto yit = training_data_y.begin();

	std::vector<double> valid_data_x(valid->data_x_.begin(), valid->data_x_.end());
	std::vector<double> valid_data_y(valid->data_y_.begin(), valid->data_y_.end());
	std::vector<double> test_data_x(test->data_x_.begin(), test->data_x_.end());
	std::vector<double> test_data_y(test->data_y_.begin(), test->data_y_.end());
	auto valid_xit = valid_data_x.begin();
	auto valid_yit = valid_data_y.begin();
	auto test_xit = test_data_x.begin();
	auto test_yit = test_data_y.begin();

	for (size_t epoch = 0; epoch < params.training_epochs_ && keep_looping; epoch++)
	{
		for (size_t mb_idx = 0; mb_idx < n_train_batches; mb_idx++)
		{
			std::vector<double> xbatch(xit + mb_idx * inbatch, xit + (mb_idx + 1) * inbatch);
			std::vector<double> ybatch(yit + mb_idx * outbatch, yit + (mb_idx + 1) * outbatch);
			finetune_in = xbatch;
			finetune_out = ybatch;
			train_update(true);

			size_t iter = (epoch - 1) * n_train_batches + mb_idx;

			if (((iter + 1) % validation_frequency) == 0)
			{
				std::vector<double> xbatch_valid(valid_xit + mb_idx * inbatch, valid_xit + (mb_idx + 1) * inbatch);
				std::vector<double> ybatch_valid(valid_yit + mb_idx * outbatch, valid_yit + (mb_idx + 1) * outbatch);
				finetune_in = xbatch_valid;
				finetune_out = ybatch_valid;

				double validate_loss = nnet::expose<double>(train_loss)[0];
				std::cout << "epoch " << epoch << ", minibatch "
					<< mb_idx + 1 << "/" << n_train_batches << ", validation error "
					<< validate_loss << std::endl;

				if (validate_loss < best_validation_loss)
				{
					// improve patience if loss improvement is good enough
					if (validate_loss < best_validation_loss * improvement_threshold)
					{
						patience = std::max(patience, iter * patience_increase);
					}

					best_validation_loss = validate_loss;
					best_iter = iter;

					std::vector<double> xbatch_test(test_xit + mb_idx * inbatch, test_xit + (mb_idx + 1) * inbatch);
					std::vector<double> ybatch_test(test_yit + mb_idx * outbatch, test_yit + (mb_idx + 1) * outbatch);
					finetune_in = xbatch_test;
					finetune_out = ybatch_test;

					double test_loss = nnet::expose<double>(train_loss)[0];
					std::cout << "\tepoch " << epoch << ", minibatch "
						<< mb_idx + 1 << "/" << n_train_batches << ", test error of best model "
						<< test_loss * 100.0 << std::endl;
				}
			}

			if (patience <= iter)
			{
				keep_looping = false;
				break;
			}
		}
	}

	std::cout << "Optimization complete with best validation score of "
		<< best_validation_loss * 100.0 << ", obtained at iteration "
		<< best_iter + 1 << ", with test performance "
		<< test_score * 100.0 << std::endl;
}

void mnist_test (xy_data* train, xy_data* valid, xy_data* test, test_params params)
{
	std::string serialpath = params.outdir_ + "/" + serialname;

	std::vector<double> training_data_x(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> training_data_y(train->data_y_.begin(), train->data_y_.end());
	std::vector<double> valid_data_x(valid->data_x_.begin(), valid->data_x_.end());
	std::vector<double> valid_data_y(valid->data_y_.begin(), valid->data_y_.end());
	std::vector<double> test_data_x(test->data_x_.begin(), test->data_x_.end());
	std::vector<double> test_data_y(test->data_y_.begin(), test->data_y_.end());

	size_t n_input = train->shape_.first;
	size_t n_output = 10;
	params.hiddens_.push_back(n_output);

	rocnnet::db_net model(n_input, params.hiddens_, "dbn_mnist_learner");

	if (params.pretrain_)
	{
		model.initialize();
		pretrain(model, n_input, training_data_x, params, "mnist");

		model.save(serialpath, "dbn_mnist_pretrain");
	}
	else
	{
		model.initialize(serialpath, "dbn_mnist_pretrain");
	}

	finetune(model, train, valid, test, n_input, n_output, params);

	model.save(serialpath, "dbn_mnist");

#ifdef CSV_RCD
if (rocnnet_record::record_status::rec_good)
{
	static_cast<rocnnet_record::csv_record*>(rocnnet_record::record_status::rec.get())->to_csv<double>();
}
#endif /* CSV_RCD */
}

std::vector<double> simple_op (std::vector<double> input)
{
	std::vector<double> output;
	for (size_t i = 0, n = input.size() / 2; i < n; ++i) {
		output.push_back((input[i] + input[n + i]) / 2);
	}
	return output;
}

void simpler_test (size_t n_train_sample, size_t n_test_sample, size_t n_in, test_params params)
{
	params.n_batch_ = std::min(params.n_batch_, n_train_sample);
	std::string serialpath = params.outdir_ + "/" + serialname;
	params.hiddens_ = { n_in, n_in, n_in / 2 };
	rocnnet::db_net model(n_in, params.hiddens_, "dbn_simple_learner");

	// generate test sample
	std::vector<double> train_samples = batch_generate(n_train_sample, n_in);
	std::vector<double> test_samples = batch_generate(n_test_sample, n_in);
	std::vector<double> train_out = simple_op(train_samples);
	std::vector<double> test_out = simple_op(test_samples);

	if (params.train_)
	{
		// pretrain
		if (params.pretrain_)
		{
			model.initialize();
			pretrain(model, n_in, train_samples, params, "demo");

			model.save(serialpath, "dbn_demo_pretrain");
		}
		else
		{
			model.initialize(serialpath, "dbn_demo_pretrain");
		}

		// finetune
		double inbatch = params.n_batch_ * n_in;
		double outbatch = inbatch / 2;
		nnet::placeholder<double> finetune_in(std::vector<size_t>{n_in, params.n_batch_}, "finetune_in");
		nnet::placeholder<double> finetune_out(std::vector<size_t>{n_in / 2, params.n_batch_}, "finetune_out");
		rocnnet::update_cost_t tuner = model.build_finetune_functions(finetune_in, finetune_out, params.training_lr_);
		nnet::variable_updater<double> train_update = tuner.first;
		size_t n_train_batches = n_train_sample / params.n_batch_;

		auto xit = train_samples.begin();
		auto yit = train_out.begin();

		for (size_t epoch = 0; epoch < params.training_epochs_; epoch++)
		{
			for (size_t mb_idx = 0; mb_idx < n_train_batches; mb_idx++)
			{
				std::vector<double> xbatch(xit + mb_idx * inbatch, xit + (mb_idx + 1) * inbatch);
				std::vector<double> ybatch(yit + mb_idx * outbatch, yit + (mb_idx + 1) * outbatch);
				finetune_in = xbatch;
				finetune_out = ybatch;
				train_update(true);
				std::cout << "epoch " << epoch << " fine tuning index " << mb_idx << std::endl;
			}
		}

		model.save(serialpath, "dbn_demo");

#ifdef CSV_RCD
if (rocnnet_record::record_status::rec_good)
{
	static_cast<rocnnet_record::csv_record*>(rocnnet_record::record_status::rec.get())->to_csv<double>();
}
#endif /* CSV_RCD */
	}
	else
	{
		model.initialize(serialpath, "dbn_demo");
	}

	// test
	nnet::placeholder<double> test_in(std::vector<size_t>{n_in}, "test_in");
	nnet::placeholder<double> expect_out(std::vector<size_t>{n_in / 2}, "expect_out");
	nnet::varptr<double> test_res = model.prop_up(nnet::varptr<double>(&test_in));
	nnet::varptr<double> test_error = nnet::reduce_mean(
		nnet::sqrt<double>(nnet::varptr<double>(&expect_out) - test_res));
	auto xit = test_samples.begin();
	auto yit = test_out.begin();
	double total_err = 0;
	for (size_t i = 0; i < n_test_sample; ++i)
	{
		std::vector<double> xbatch(xit + i * n_in, xit + (i + 1) * n_in);
		std::vector<double> ybatch(yit + i * n_in / 2, yit + (i + 1) * n_in / 2);
		test_in = xbatch;
		expect_out = ybatch;

		double test_err = nnet::expose<double>(test_error)[0];
		total_err += test_err;
		std::cout << "test error at " << i << ": " << test_err << std::endl;
	}
	std::cout << "total error " << total_err << std::endl;
}

int main (int argc, char** argv)
{
	// todo: replace with boost flags
	test_params params;
	std::experimental::optional<size_t> seed;
#ifdef __GNUC__ // use this gnu parser, since boost is too big for free-tier platforms
	int c;
	bool test_mnist = false;
	size_t n_simple_samples = 300;
	while ((c = getopt (argc, argv, "s:o:p:E:e:k:t:m:n:")) != -1)
	{
		switch(c)
		{
			case 's':
				seed = atoi(optarg);
				break;
			case 'o': // output directory
				params.outdir_ = std::string(optarg);
				break;
			case 'p':
				params.pretrain_ = false;
				break;
			case 'E': // epoch pretraining iteration
				params.pretrain_epochs_ = atoi(optarg);
				break;
			case 'e': // epoch training iteration
				params.training_epochs_ = atoi(optarg);
				break;
			case 'k': // k-CD or k-PCD
				params.n_cont_div_ = atoi(optarg);
				break;
			case 't':
				params.train_ = false;
				break;
			case 'm':
				test_mnist = true;
				break;
			case 'n':
				n_simple_samples = std::min(6, atoi(optarg));
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
		params.training_epochs_ = atoi(argv[2]);
	}
#endif
	if (seed)
	{
		rnd_device.seed(*seed);
		nnutils::seed_generator(*seed);
	}

	if (test_mnist)
	{
		try
		{
			Py_Initialize();
			np::initialize();
			std::vector<xy_data*> datasets = get_mnist_data();

			xy_data* training_set = datasets[0];
			xy_data* valid_set = datasets[1];
			xy_data* testing_set = datasets[2];

			mnist_test(training_set, valid_set, testing_set, params);

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
		params.pretrain_epochs_ = 1;
		params.training_epochs_ = 1;
		simpler_test(n_simple_samples, n_simple_samples / 6, 10, params);
	}

	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}
