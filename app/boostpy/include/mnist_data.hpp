//
// Created by Mingkai Chen on 2017-09-02.
//

#include <boost/python/numpy.hpp>
#include "embedding.hpp"

namespace np = bp::numpy;

#pragma once
#ifndef ROCBOOST_MNIST_DATA_HPP
#define ROCBOOST_MNIST_DATA_HPP

struct xy_data
{
	std::vector<float> data_x_;
	std::vector<float> data_y_; // Y has X's height and width 1

	std::pair<size_t, size_t> shape_; // shape_[0] represents X's width, shape_[1] is X's height
};

std::vector<xy_data*> get_mnist_data (void);

#endif /* ROCBOOST_MNIST_DATA_HPP */
