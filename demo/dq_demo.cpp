// test dq_demo on c++ side

#include "dq_net.hpp"

int main (int argc, char** argv)
{
	std::string outdir = ".";
	if (argc > 1)
	{
		outdir = std::string(argv[1]);
	}
	std::string serialname = "dqntest.pbx";
	std::string serialpath = outdir + "/" + serialname;

	size_t n_in = 10;
	size_t n_out = 5;
	size_t n_batch = 3;
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(9, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(8, nnet::sigmoid<double>),
		rocnnet::IN_PAIR(n_out, nnet::sigmoid<double>)
	};
	nnet::vgb_updater bgd;
	bgd.learning_rate_ = 0.9;
	rocnnet::dq_net untrained_dqn(n_in, hiddens, bgd);
	rocnnet::dq_net* trained_dqn = new rocnnet::dq_net(untrained_dqn, "trained_dqn");
	rocnnet::dq_net pretrained_dqn(n_in, hiddens, bgd);
	untrained_dqn.initialize();
	trained_dqn->initialize();
	pretrained_dqn.initialize(serialpath);
	
	// that's it so far
	
	delete trained_dqn;

	return 0;
}
