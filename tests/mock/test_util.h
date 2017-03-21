//
// Created by Mingkai Chen on 2016-11-25.
//

#include <iostream>
#include "tensor/tensor.hpp"

#pragma once
#ifndef ROCNNET_TEST_UTIL_H
#define ROCNNET_TEST_UTIL_H


// tensors equal according to specs
bool tensorshape_equal (
	const nnet::tensorshape& ts1,
	const nnet::tensorshape& ts2);

void print (std::vector<double> raw);

void print_tensor (nnet::tensor<double>* t);


#endif //ROCNNET_TEST_UTIL_H
