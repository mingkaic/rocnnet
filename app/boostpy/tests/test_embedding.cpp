#include <iostream>
#include "mnist_data.hpp"

int main (int argc, char** argv)
{
	try
	{
		Py_Initialize();
		np::initialize();

		std::vector<xy_data*> datasets = get_mnist_data();

		for (xy_data* dataset : datasets)
		{
			delete dataset;
		}

		return 0;
	}
	catch(const bp::error_already_set&)
	{
		std::cerr << ">>> Error! Uncaught exception:\n";
		PyErr_Print();
		return 1;
	}
}
