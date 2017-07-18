//
// Created by Mingkai Chen on 2017-05-06.
//

#include "dqn_agent.hpp"

#ifdef ROCNNET_DQN_AGENT_HPP


#include "models/dq_net.hpp"


dqn_agent::dqn_agent (unsigned int n_input,
	std::vector<unsigned int> hiddensizes,
	std::vector<unsigned int> nonlinearities,
	double learning_rate,
	double decay,
	double random_action_probability,
	double exploration_period,
	unsigned int store_every_nth,
	unsigned int train_every_nth,
	unsigned int minibatch_size,
	double discount_rate,
	double max_experience,
	double target_network_update_rate)
{
	std::vector<rocnnet::IN_PAIR> hiddens;
	size_t n = hiddensizes.size();
	assert(n == nonlinearities.size());
	for (size_t i = 0; i < n; i++)
	{
		rocnnet::VAR_FUNC act;
		switch(nonlinearities[i])
		{
			case SIGMOID:
				act = nnet::sigmoid<double>;
				break;
			case TANH:
				act = nnet::tanh<double>;
				break;
			case IDENTITY:
				act = nnet::identity<double>;
				break;
		}
		hiddens.push_back({hiddensizes[i], act});
	}

	nnet::rmspropupdater learner(learning_rate, decay);

	rocnnet::dqn_param param;
	param.train_interval_ = train_every_nth;
	param.rand_action_prob_ = random_action_probability;
	param.discount_rate_ = discount_rate;
	param.target_update_rate_ = target_network_update_rate;
	param.exploration_period_ = exploration_period;
	param.store_interval_ = store_every_nth;
	param.mini_batch_size_ = minibatch_size;
	param.max_exp_ = max_experience;

	rocnnet::mlp* brain = new rocnnet::mlp(n_input, hiddens);
	brain_ = new rocnnet::dq_net(brain, learner, param, "pynet");
}


dqn_agent::~dqn_agent (void)
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	delete netbrain;
}


unsigned int dqn_agent::action (std::vector<double> input)
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	return (unsigned int) netbrain->action(input);
}


void dqn_agent::store (std::vector<double> observations,
	unsigned int action_idx,
	double reward,
	std::vector<double> new_obs)
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	netbrain->store(observations, action_idx, reward, new_obs);
}


void dqn_agent::train (void)
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	netbrain->train();
}


void dqn_agent::initialize (std::string ifile)
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	netbrain->initialize(ifile);
}


bool dqn_agent::save (std::string ofile) const
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	return netbrain->save(ofile);
}


#endif
