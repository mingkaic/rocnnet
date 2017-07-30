//
// Created by Mingkai Chen on 2017-03-12.
//

#include "graph/inode.hpp"
#include "graph/leaf/ileaf.hpp"
#include "graph/leaf/ivariable.hpp"
#include "graph/leaf/constant.hpp"
#include "graph/leaf/variable.hpp"
#include "graph/leaf/placeholder.hpp"
#include "graph/connector/iconnector.hpp"
#include "graph/connector/immutable/immutable.hpp"
#include "graph/varptr.hpp"

namespace nnet
{

template class varptr<double>;

template class placeptr<double>;

template class constant<double>;

template class variable<double>;

template class placeholder<double>;

template class immutable<double>;

template class immutable<unsigned>;

template class merged_immutable<double>;

}