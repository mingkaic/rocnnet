// test dq_demo on c++ side

#include "dq_net.hpp"
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
	std::string serialname = "dqntest.pbx";
	std::string serialpath = outdir + "/" + serialname;

	size_t episode_count = 250;
	size_t max_steps = 100; 
	size_t n_in = 4;
	size_t n_out = 2;
	size_t n_batch = 3;
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(3, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_out, nnet::sigmoid<double>)
	};
	nnet::vgb_updater bgd;
	bgd.learning_rate_ = 0.9;
	rocnnet::dqn_param param;
	param.mini_batch_size_ = 10;
	
	rocnnet::dq_net untrained_dqn(n_in, hiddens, bgd, param, "untrained_dqn");
	untrained_dqn.initialize();

	rocnnet::dq_net* trained_dqn = new rocnnet::dq_net(untrained_dqn, "trained_dqn");
	trained_dqn->initialize();

	rocnnet::dq_net pretrained_dqn(n_in, hiddens, bgd, param, "pretrained_dqn");
	pretrained_dqn.initialize(serialpath);
	
	// action and observations are randomly generated, so they're meaningless. 
	// we're evaluating whether we hit any assertions/exceptions
	int exit_code = 0;
	try
	{
		std::vector<double> observations;
		std::vector<double> new_observations;
		std::vector<double> expect_out;
		std::vector<double> output;
		size_t action;
		double reward;
		size_t n_observations = 4;
		size_t n_actions = 2;
		for (size_t i = 0; i < episode_count; i++)
		{
			observations = batch_generate(n_observations, 1);
			expect_out = avgevry2(observations);
			reward = 0;
			for (size_t j = 0; j < max_steps; j++)
			{
				output = trained_dqn->direct_out(observations);
				action = output[0] > output[1] ? 0 : 1;
				double err = 0;
				for (size_t i = 0; i < n_actions; i++)
				{
					err += std::abs(output[i] - expect_out[i]);
				}
				err /= n_actions;
				std::cout << "error " << err << " turn " << j << std::endl;
				reward = 2*(1.0 - err) - 1;
				
				new_observations = batch_generate(n_observations, 1);
				expect_out = avgevry2(observations);

				trained_dqn->store(observations, action, reward, new_observations);
				trained_dqn->train();

				observations = new_observations;
			}
			std::cout << "episode " << i << " trained " << trained_dqn->get_numtrained() << " times" << std::endl;
		}
		exit_code = 0;
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		exit_code = 1;
	}

#ifdef EDGE_RCD
	rocnnet_record::erec::rec.to_csv<double>();
#endif /* EDGE_RCD */
	
	delete trained_dqn;

	return exit_code;
}
