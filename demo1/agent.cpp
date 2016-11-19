
#include "dqn.hpp"
#include "graph/functions.hpp"

int main (void)
{
    const size_t n_observation = 10;
    const size_t n_hidden = 10;
    const size_t n_actions = 8;
    nnet::ioptimizer<double>* optimizer = new gd_optimizer(0.001);
    nnet::dq_net model(
        bot_sight,
		std::vector<nnet::IN_PAIR>{
        	IN_PAIR(n_hidden, nnet::sigmoid<double>),
        	IN_PAIR(n_hidden, nnet::sigmoid<double>),
        	IN_PAIR(n_actions, nnet::sigmoid<double>),
        },
		optimizer,
		4, // training interval
		0.1, // random action prob
		0.99, // discount factor
		0.01, // update rate
		4, // store interval
		8, // mini_batch_size
		10000 // max_exp
	);
	
	// do something with the model
    
    delete optimizer;
    return 0;
}