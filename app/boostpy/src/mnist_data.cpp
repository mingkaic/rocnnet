//
// Created by Mingkai Chen on 2017-09-02.
//

#include "mnist_data.hpp"
#include <iostream>

#ifdef ROCBOOST_MNIST_DATA_HPP

template <typename T>
std::vector<T> to_vector (np::ndarray narr, std::vector<size_t>& shape)
{
	shape.clear();
	if (narr.is_none())
	{
		return {};
	}
	np::dtype type = narr.get_dtype();
	np::dtype expectt = np::dtype::get_builtin<T>();
	size_t n_dim = narr.get_nd();
	Py_intptr_t const* ndshape = narr.get_shape();
	size_t total_n = 1;
	for (size_t i = 0; i < n_dim; i++)
	{
		size_t sval = ndshape[i];
		shape.push_back(sval);
		total_n *= sval;
	}

	char* cdata = narr.get_data();
	T* data = (T*) cdata;
	std::vector<T> result;

	for (size_t i = 0; i < total_n; i++)
	{
		result.push_back(data[i]);
	}

	return result;
}

// heap allocate xy_data, does not own
xy_data* get_xy_data(bp::tuple xy_set)
{
	np::ndarray data_set_x = bp::extract<np::ndarray>(xy_set[0]);
	np::ndarray data_set_y = bp::extract<np::ndarray>(xy_set[1]);

	std::vector<size_t> shape;
	std::vector<float> data_x = to_vector<float>(data_set_x, shape);
	assert(shape.size() == 2); // x should be a 2d array
	size_t shape_y = shape[0];
	size_t shape_x = shape[1];
	std::vector<float> data_y = to_vector<float>(data_set_y, shape);
	assert(shape.size() == 1); // y should be a 1d array
	assert(shape_y == shape[0]);

	return new xy_data{data_x, data_y, std::pair<size_t, size_t>{shape_x, shape_y}};
}

std::vector<xy_data*> get_mnist_data (void)
{
	std::string mnist = "mnist.pkl.gz";
	bp::tuple dataset = pickle(mnist);

	size_t n_dataset = bp::len(dataset);
	assert(n_dataset == 3);

	// assert dataset has format <training_set, valid_set, test_set>
	bp::tuple training_set = bp::extract<bp::tuple>(dataset[0]);
	bp::tuple valid_set = bp::extract<bp::tuple>(dataset[1]);
	bp::tuple test_set = bp::extract<bp::tuple>(dataset[2]);

	size_t n_dtraining = bp::len(training_set);
	size_t n_dvalid = bp::len(valid_set);
	size_t n_dtest = bp::len(test_set);
	assert(n_dtraining == 2 && n_dvalid == 2 && n_dtest == 2);

	xy_data* training = get_xy_data(training_set);
	xy_data* valid = get_xy_data(valid_set);
	xy_data* testing = get_xy_data(test_set);

	return {training, valid, testing};
}

#endif
