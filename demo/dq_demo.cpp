// test dq_demo on c++ side

#include "dq_net.hpp"

static std::default_random_engine generator;

std::vector<double> generate_observation (size_t n_obs)
{
	std::uniform_real_distribution<double> dis(-1, 1);
	std::vector<double> output(n_obs);
	std::generate(output.begin(), output.end(), [&dis](){ return dis(generator); });
	return output;
}

double generate_reward (void)
{
	std::uniform_real_distribution<double> dis(0, 2);
	return dis(generator);
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
	rocnnet::dq_net* trained_dqn = new rocnnet::dq_net(untrained_dqn, "trained_dqn");
	rocnnet::dq_net pretrained_dqn(n_in, hiddens, bgd, param, "pretrained_dqn");
	untrained_dqn.initialize();
	trained_dqn->initialize();
	pretrained_dqn.initialize(serialpath);
	
	// action and observations are randomly generated, so they're meaningless. 
	// we're evaluating whether we hit any assertions/exceptions
	std::vector<double> observations;
	std::vector<double> new_observations;
	double reward;
	size_t action;
	size_t n_observations = 4;
	size_t n_actions = 2;
	for (size_t i = 0; i < episode_count; i++)
	{
		observations = generate_observation(n_observations);
		reward = 0;
		for (size_t j = 0; j < max_steps; j++)
		{
			action = trained_dqn->action(observations);
			new_observations = generate_observation(n_observations);
			reward = generate_reward();
			
			trained_dqn->store(observations, action, reward, new_observations);
			trained_dqn->train();
			
			observations = new_observations;
		}
	}
	
	delete trained_dqn;

	return 0;
}
