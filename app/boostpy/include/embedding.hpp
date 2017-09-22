//
// Created by Mingkai Chen on 2017-09-02.
//

#include <vector>
#include <boost/python.hpp>

namespace bp = boost::python;

#pragma once
#ifndef ROCBOOST_EMBEDDING_HPP
#define ROCBOOST_EMBEDDING_HPP

bp::tuple pickle (std::string data_path);

void mnist_imageout (std::vector<std::vector<double> >& imgdata,
	std::vector<size_t> inner_dims, size_t n_chains, size_t n_samples);

#endif /* ROCBOOST_EMBEDDING_HPP */
