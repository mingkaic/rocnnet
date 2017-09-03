//
// Created by Mingkai Chen on 2017-09-02.
//

#include <boost/python.hpp>

namespace bp = boost::python;

#ifndef ROCBOOST_EMBEDDING_HPP
#define ROCBOOST_EMBEDDING_HPP

bp::tuple pickle(std::string data_path);

#endif //ROCBOOST_EMBEDDING_HPP
