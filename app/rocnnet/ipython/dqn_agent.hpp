//
// Created by Mingkai Chen on 2017-05-06.
//

#ifndef ROCNNET_DQN_AGENT_HPP
#define ROCNNET_DQN_AGENT_HPP


#include <vector>
#include <string>


enum NL
{
	SIGMOID=0,
	TANH=1,
	IDENTITY=2
};


struct dqn_agent
{
	// builds dq_net using RMSProp
	dqn_agent (unsigned int n_input, 
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
		double target_network_update_rate);

	~dqn_agent (void);

	unsigned int action (std::vector<double> input);

	void store (std::vector<double> observations,
		unsigned int action_idx,
		double reward,
		std::vector<double> new_obs);

	void train (void);

	void initialize (std::string ifile);

	bool save (std::string ofile) const;

private:
	void* brain_;
};


#endif //ROCNNET_DQN_AGENT_HPP
