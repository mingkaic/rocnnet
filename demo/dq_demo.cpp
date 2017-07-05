// test dq_demo on c++ side

#include "dq_net.hpp"
#include "edgeinfo/comm_record.hpp"

static std::default_random_engine rnd_device(std::time(NULL));

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

// calculates the circumference distance between A and B assuming A and B represent positions on a circle with circumference wrap_size
inline size_t wrapdist (size_t A, size_t B, size_t wrap_size)
{
	double within_dist = std::min(A - B, B - A);
	double edge_dist = std::min(A + wrap_size - B, B + wrap_size - A);
	return std::min(within_dist, edge_dist);
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
//	size_t max_steps = 5000;
	size_t max_steps = 100;
	size_t n_observations = 10;
	size_t n_actions = 9;
	std::vector<rocnnet::IN_PAIR> hiddens = {
		// use same sigmoid in static memory once deep copy is established
		rocnnet::IN_PAIR(9, nnet::tanh<double>),
		rocnnet::IN_PAIR(n_actions, nnet::identity<double>)
	};
	nnet::rmspropupdater bgd;
	bgd.learning_rate_ = 0.1;
	rocnnet::dqn_param param;
	param.store_interval_ = 1;
	param.discount_rate_ = 0.99;
	param.exploration_period_ = 0;

	rocnnet::ml_perceptron* brain = new rocnnet::ml_perceptron(n_observations, hiddens);
	rocnnet::dq_net untrained_dqn(brain, bgd, param, "untrained_dqn");
	untrained_dqn.initialize();
	rocnnet::dq_net trained_dqn(untrained_dqn, "trained_dqn");
//	trained_dqn->initialize(serialpath, "trained_dqn");

	rocnnet::dq_net pretrained_dqn(untrained_dqn, "pretrained_dqn");
	pretrained_dqn.initialize(serialpath, "trained_dqn");

	int exit_status = 0;
	// exit code:
	//	0 = fine
	//	1 = overfitting
	//	2 = training error rate is wrong
	std::vector<double> observations;
	std::vector<double> new_observations;
	std::vector<double> expect_out;
	std::list<double> error_queue;
	size_t err_queue_size = 10;
	size_t action_dist = n_actions / 2;
	for (size_t i = 0; i < episode_count; i++)
	{
		std::vector<double> output;
		double avgreward = 0;
		observations = batch_generate(n_observations, 1);
		expect_out = avgevry2(observations);
		double episode_err = 0;
		for (size_t j = 0; j < max_steps; j++)
		{
			size_t action = trained_dqn.action(observations);
			auto mit = std::max_element(expect_out.begin(), expect_out.end());
			size_t expect_action = std::distance(expect_out.begin(), mit);

			// err = [0, 1, 2, 3]
			double err = wrapdist(expect_action, action, n_actions);

			double reward = 1 - 2.0 * err / action_dist;
			avgreward += reward;

			new_observations = batch_generate(n_observations, 1);
			expect_out = avgevry2(new_observations);

			trained_dqn.store(observations, action, reward, new_observations);
			trained_dqn.train();

			observations = new_observations;
			episode_err += err / action_dist;
		}
		avgreward /= max_steps;
		episode_err /= max_steps;

		error_queue.push_back(episode_err);
		if (error_queue.size() > err_queue_size)
		{
			error_queue.pop_front();
		}

		// allow ~15% decrease in accuracy (15% increase in error) since last episode
		// otherwise declare that we overfitted and quit
		double avgerr = 0;
		for (double last_error : error_queue)
		{
			avgerr += last_error;
		}
		avgerr /= error_queue.size();
		if (avgerr - episode_err > 0.15)
		{
			std::cout << "uh oh, we hit a snag, we shouldn't save for this round" << std::endl;
			exit_status = 1;
		}

		if (std::isnan(episode_err)) throw std::exception();
		std::cout << "episode " << i << " performance: " << episode_err * 100 << "% average error, reward: " << avgreward << std::endl;
	}

	std::vector<double> untrained_output;
	std::vector<double> trained_output;
	std::vector<double> pretrained_output;
	double total_untrained_err = 0;
	double total_trained_err = 0;
	double total_pretrained_err = 0;
	for (size_t j = 0; j < max_steps; j++)
	{
		observations = batch_generate(n_observations, 1);
		expect_out = avgevry2(observations);

		double untrained_action = untrained_dqn.never_random(observations);
		double trained_action = trained_dqn.never_random(observations);
		double pretrained_action = pretrained_dqn.never_random(observations);

		auto mit = std::max_element(expect_out.begin(), expect_out.end());
		size_t expect_action = std::distance(expect_out.begin(), mit);

		double untrained_err = wrapdist(expect_action, untrained_action, n_actions);
		double trained_err = wrapdist(expect_action, trained_action, n_actions);
		double pretrained_err = wrapdist(expect_action, pretrained_action, n_actions);

		total_untrained_err += untrained_err / action_dist;
		total_trained_err += trained_err / action_dist;
		total_pretrained_err += pretrained_err / action_dist;
	}
	total_untrained_err /= max_steps;
	total_trained_err /= max_steps;
	total_pretrained_err /= max_steps;
	std::cout << "untrained performance: " << total_untrained_err * 100 << "% average error" << std::endl;
	std::cout << "trained performance: " << total_trained_err * 100 << "% average error" << std::endl;
	std::cout << "pretrained performance: " << total_pretrained_err * 100 << "% average error" << std::endl;

	if (total_untrained_err < total_trained_err) exit_status = 2;

	if (exit_status == 0)
	{
		trained_dqn.save(serialname);
	}

#ifdef EDGE_RCD
if (rocnnet_record::erec::rec_good)
	rocnnet_record::erec::rec.to_csv<double>(
		static_cast<nnet::iconnector<double>*>(trained_dqn.get_trainout().get()));
#endif /* EDGE_RCD */

	return exit_status;
}
