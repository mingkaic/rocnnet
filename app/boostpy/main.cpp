#include <iostream>
#include "embedding.hpp"

int main()
{
	try
	{
		Py_Initialize();

		std::string mnist = "mnist.pkl.gz";
		bp::tuple dataset = pickle(mnist);

		// assert dataset has format <training_set, valid_set, test_set>
		auto training_set = boost::get<0>(dataset[0]);
		auto valid_set = boost::get<1>(dataset[0]);
		auto test_set = boost::get<2>(dataset[0]);

		return 0;
	}
	catch(const bp::error_already_set&)
	{
		std::cerr << ">>> Error! Uncaught exception:\n";
		PyErr_Print();
		return 0;
	}
}
