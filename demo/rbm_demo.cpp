//
// Created by Mingkai Chen on 2017-07-19.
//

#include "compounds/rbm.hpp"

int main (int argc, char** argv)
{
	// using test from http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
	// <Harry Potter, Avatar, LOTR, Gladiator, Titanic, Glitter>
	std::vector<double> training_data = {
		1,1,1,0,0,0, // Alice likes
		1,0,1,0,0,0, // Bob
		1,1,1,0,0,0, // Carol
		0,0,1,1,1,0, // David
		0,0,1,1,0,0, // Eric
		0,0,1,1,1,0 // Fred
	};
	std::vector<double> test_data = {0,0,0,1,1,0};

	size_t n_input = 6;
	size_t n_epoch = 5000;
	double learning_rate = 0.1;
	rocnnet::IN_PAIR hiddens = {2, nnet::sigmoid<double>};
	rocnnet::rbm model(n_input, hiddens, "movie_rater");
	model.initialize();

	rocnnet::fit(model, training_data, {n_epoch, 1, learning_rate});

	nnet::placeholder<double> test_in(std::vector<size_t>{n_input});
	nnet::varptr<double> test_out = model(&test_in);
	test_in = test_data;
	std::vector<double> out = nnet::expose<double>(test_out);
	for (double o : out)
	{
		std::cout << o << ", ";
	}
	std::cout << std::endl;

	return 0;
}
