//
// Created by Mingkai Chen on 2017-05-06.
//

#ifndef ROCNNET_DQN_AGENT_HPP
#define ROCNNET_DQN_AGENT_HPP


#include <vector>


struct dqn_agent
{
	// builds dq_net using sigmoid activations, vanilla gradient descent and default parameters
	dqn_agent (unsigned int n_input,
		std::vector<unsigned int> nhiddens,
		double learning_rate,
		std::string name);

	~dqn_agent (void);

	std::vector<double> action (std::vector<double>& input);

	void store (std::vector<double> observation,
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
