//
// Created by Mingkai Chen on 2016-10-20.
//

#ifndef CNNET_UTILS_HPP
#define CNNET_UTILS_HPP

#include <stdio.h>
#include <algorithm>
#include <memory>
#include "tenncor/tenncor.hpp"
#include "gtest/gtest.h"

// TODO: standardize these by implementing factories
#define MAKE_PLACEHOLDER(shape, name) std::make_shared<nnet::placeholder<double> >(shape, name)
#define MAKE_VARIABLE(shape, name) std::make_shared<nnet::variable<double> >(shape, name)
#define EXPOSE(var) std::make_shared<nnet::expose<double> >(var)
#define EXPOSE_PTR std::shared_ptr<nnet::expose<double> >

#endif //CNNET_UTILS_HPP
