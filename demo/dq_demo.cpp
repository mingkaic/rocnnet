// test dq_demo on c++ side

#include "dq_net.hpp"
#include "edgeinfo/comm_record.hpp"

static std::default_random_engine rnd_device;//(std::time(NULL));

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
	std::string outdir = ".";
	if (argc > 1)
	{
		outdir = std::string(argv[1]);
	}
	std::string serialname = "dqn_test.pbx";
	std::string serialpath = outdir + "/" + serialname;

	size_t episode_count = 250;
	size_t max_steps = 100;
	size_t n_observations = 10;
	size_t n_actions = 5;
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(9, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_actions, nnet::sigmoid<double>)
	};
	nnet::vgb_updater bgd;
	bgd.learning_rate_ = 0.9;
	rocnnet::dqn_param param;
	param.mini_batch_size_ = 10;
	param.max_exp_ = 100;
	
	rocnnet::dq_net untrained_dqn(n_observations, hiddens, bgd, param, "untrained_dqn");
	untrained_dqn.initialize();

	rocnnet::dq_net* trained_dqn = new rocnnet::dq_net(untrained_dqn, "trained_dqn");
	trained_dqn->initialize(serialpath);

	// action and observations are randomly generated, so they're meaningless. 
	// we're evaluating whether we hit any assertions/exceptions
	int exit_code = 0;
	// exit code:
	//	0 = fine
	//	1 = internal error
	//	2 = overfitting
	try
	{
		std::vector<double> observations;
		std::vector<double> new_observations;
		std::vector<double> expect_out;
		std::vector<double> output;
		size_t action;
		double reward;
		double last_episode_error = 1;
		for (size_t i = 0; i < episode_count; i++)
		{
			observations = batch_generate(n_observations, 1);
			expect_out = avgevry2(observations);
			reward = 0;
			double episode_err = 0;
			for (size_t j = 0; j < max_steps; j++)
			{
				output = trained_dqn->direct_out(observations);
				auto mit = (std::max_element(output.begin(), output.end()));
				action = std::distance(output.begin(), mit);
				double err = output[action] - expect_out[action];
				reward = std::round(2*(1.0 - err) - 1);

				new_observations = batch_generate(n_observations, 1);
				expect_out = avgevry2(observations);

				trained_dqn->store(observations, action, reward, new_observations);
				trained_dqn->train();

				observations = new_observations;
				episode_err += err;
			}
			episode_err /= max_steps;
			// allow ~15% decrease in accuracy (15% increase in error) since last episode
			// otherwise declare that we overfitted and quit
			if (episode_err - last_episode_error > 0.2)
			{
				std::cout << "uh oh, we hit a snag, we shouldn't save for this round" << std::endl;
				exit_code = 2;
			}

			if (std::isnan(episode_err)) throw std::exception();
			std::cout << "episode " << i << " performance: " << episode_err * 100 << "% average error" << std::endl;
		}

		double total_untrained_err = 0;
		double total_trained_err = 0;
		for (size_t j = 0; j < max_steps; j++)
		{
			observations = batch_generate(n_observations, 1);
			output = untrained_dqn.direct_out(observations);
			std::vector<double> train_output = trained_dqn->direct_out(observations);
			auto mit = (std::max_element(output.begin(), output.end()));
			action = std::distance(output.begin(), mit);
			double untrained_err = std::abs(output[action] - expect_out[action]);

			mit = (std::max_element(train_output.begin(), train_output.end()));
			action = std::distance(train_output.begin(), mit);
			double trained_err = std::abs(train_output[action] - expect_out[action]);

			total_untrained_err += untrained_err;
			total_trained_err += trained_err;
		}
		total_untrained_err /= max_steps;
		total_trained_err /= max_steps;

		std::cout << "untrained performance: " << total_untrained_err * 100 << "% average error" << std::endl;
		std::cout << "trained performance: " << total_trained_err * 100 << "% average error" << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		exit_code = 1;
	}

#ifdef EDGE_RCD
	rocnnet_record::erec::rec.to_csv<double>();
#endif /* EDGE_RCD */

	if (exit_code == 0)
	{
		trained_dqn->save(serialname);
	}
	
	delete trained_dqn;

	return exit_code;
}
