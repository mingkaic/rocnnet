//
// Created by Mingkai Chen on 2017-09-21.
//

#include "models/db_net.hpp"
#include "mnist_data.hpp"
#include "edgeinfo/comm_record.hpp"

#ifdef __GNUC__
#include <unistd.h>
#endif

struct test_params
{
	size_t pretrain_epochs_ = 100;
	double pretrain_lr_ = 0.01;

	size_t training_epochs_ = 1000;
	double training_lr_ = 0.1;

	bool pretrain_ = true;

	size_t n_cont_div_ = 15;
	size_t n_hidden_ = 50;
	size_t n_batch_ = 20;
	size_t n_test_chain_ = 20;
	size_t n_samples_ = 10;
	std::string outdir_ = ".";
	bool train_ = true;
};

std::string serialname = "dbn_test.pbx";

void pretrain (rocnnet::db_net& model, size_t n_input,
	std::vector<double> data, test_params params)
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
				mean_cost += nnet::expose<double>(cost)[0] / n_train_batches;
			}
			std::cout << "pre-trained layer " << pidx << " epoch "
				<< e << " has mean cost " << mean_cost << std::endl;

			model.save(serialpath, "dbn_demo_prerain"); // save in case of problems
		}
	}

#ifdef EDGE_RCD
	if (rocnnet_record::erec::rec_good)
	{
		rocnnet_record::erec::rec.to_csv<double>();
	}
#endif /* EDGE_RCD */
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

	size_t n_train_batches = train->shape_.first;
	size_t n_input = 28 * 28;
	size_t n_output = 10;
	std::vector<size_t> hiddens = { 1000, 1000, 1000, n_output };

	rocnnet::db_net model(n_input, hiddens, "dbn_mnist_learner");

	if (params.pretrain_)
	{
		model.initialize();
		pretrain(model, n_input, training_data_x, params);

		model.save(serialpath, "dbn_demo_prerain");
	}
	else
	{
		model.initialize(serialpath, "dbn_demo_prerain");
	}

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

	double inbatch = params.n_batch_ * n_input;
	auto xit = training_data_x.begin();
	auto yit = training_data_y.begin();
	auto valid_xit = valid_data_x.begin();
	auto valid_yit = valid_data_y.begin();
	auto test_xit = test_data_x.begin();
	auto test_yit = test_data_y.begin();
	for (size_t epoch = 0; epoch < params.training_epochs_ && keep_looping; epoch++)
	{
		for (size_t mb_idx = 0; mb_idx < n_train_batches; mb_idx++)
		{
			std::vector<double> xbatch(xit + mb_idx * inbatch, xit + (mb_idx + 1) * inbatch);
			std::vector<double> ybatch(yit + mb_idx * inbatch, yit + (mb_idx + 1) * inbatch);
			finetune_in = xbatch;
			finetune_out = ybatch;
			train_update(true);

			size_t iter = (epoch - 1) * n_train_batches + mb_idx;

			if (((iter + 1) % validation_frequency) == 0)
			{
				std::vector<double> xbatch_valid(valid_xit + mb_idx * inbatch, valid_xit + (mb_idx + 1) * inbatch);
				std::vector<double> ybatch_valid(valid_yit + mb_idx * inbatch, valid_yit + (mb_idx + 1) * inbatch);
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
					std::vector<double> ybatch_test(test_yit + mb_idx * inbatch, test_yit + (mb_idx + 1) * inbatch);
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

	model.save(serialpath, "dbn_demo");

#ifdef EDGE_RCD
	if (rocnnet_record::erec::rec_good)
	{
		rocnnet_record::erec::rec.to_csv<double>();
	}
#endif /* EDGE_RCD */
}

int main (int argc, char** argv)
{
	// todo: replace with boost flags
	test_params params;
	std::experimental::optional<size_t> seed;
#ifdef __GNUC__ // use this gnu parser, since boost is too big for free-tier platforms
	int c;
	while ((c = getopt (argc, argv, "s:o:p:E:e:k:t:")) != -1)
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
		nnutils::seed_generator(*seed);
	}

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

	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}
