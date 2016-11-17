//
// Created by Mingkai Chen on 2016-11-15.
//

#ifndef ROCNNET_MOCK_CONSTANT_H
#define ROCNNET_MOCK_CONSTANT_H

#include "gmock/gmock.h"
#include "graph/variable/constant.hpp"

class mock_constant : public nnet::constant<double>
{
    
};

#endif//ROCNNET_MOCK_CONSTANT_H