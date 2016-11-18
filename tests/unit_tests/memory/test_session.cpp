//
// Created by Mingkai Chen on 2016-11-17.
//

#include "gtest/gtest.h"
#include "memory/session.hpp"
#include "graph/variable/variable.hpp"
#include "graph/variable/placeholder.hpp"
#include "graph/variable/constant.hpp"


TEST(SESSION, Registration)
{
	nnet::session& sess = nnet::session::get_instance();

	nnet::variable<double> var1(1);
	nnet::placeholder<double> var2(std::vector<size_t>{1});
	nnet::constant<double>* var3 = nnet::constant<double>::build(1);

	delete var3;
}