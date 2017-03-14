//
// Created by Mingkai Chen on 2017-03-12.
//

#include "tensor/tensor.hpp"
#include "tensor/tensor_handler.hpp"

namespace nnet
{

// Instantiate tensors for instrumentation
template class itensor_handler<double>;

template class transfer_func<double>;

template class const_init<double>;

template class rand_uniform<double>;

template class tensor<double>;

}
