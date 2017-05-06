//
// Created by Mingkai Chen on 2017-05-06.
//

#include "dqn_agent.hpp"

#ifdef ROCNNET_DQN_AGENT_HPP


#include "dq_net.hpp"


dqn_agent::dqn_agent (unsigned int n_input,
	std::vector<unsigned int> hiddensizes,
	double learning_rate,
	std::string name)
{
	std::vector<rocnnet::IN_PAIR> hiddens;
	for (unsigned hid_size : hiddensizes)
	{
		hiddens.push_back({hid_size, nnet::sigmoid<double>});
	}
	nnet::vgb_updater bgd;
	bgd.learning_rate_ = learning_rate;
	rocnnet::dqn_param param;
	brain_ = new rocnnet::dq_net(n_input, hiddens, bgd, param, name);
}


dqn_agent::~dqn_agent (void)
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	delete netbrain;
}


std::vector<double> dqn_agent::action (std::vector<double> input)
{
	rocnnet::dq_net* netbrain = (rocnnet::dq_net*)brain_;
	return netbrain->action(input);
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
