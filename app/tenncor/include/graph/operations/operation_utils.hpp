/*!
 *
 *  operations_util.hpp
 *  cnnet
 *
 *  Purpose:
 *  shared utility functions for all operation functions
 *  also useful for debugging purposes
 *
 *  Created by Mingkai Chen on 2017-09-07.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/inode.hpp"

#pragma once
#ifndef TENNCOR_OPERATION_UTILS_HPP
#define TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

//! return null if no such parent named opname is found, otherwise return parent
template <typename T>
inode<T>* unary_parent_search (inode<T>* operand, std::string opname);

//! return null if no such parent satisfies both operands is found, otherwise return parent
template <typename T>
inode<T>* ordered_binary_parent_search (inode<T>* a, inode<T>* b, std::string opname);

//! return null if no such parent satisfies both operands is found, otherwise return parent
template <typename T>
inode<T>* unordered_binary_parent_search (inode<T>* a, inode<T>* b, std::string opname);

}

#include "../../../src/graph/operations/operation_utils.ipp"

#endif /* TENNCOR_OPERATION_UTILS_HPP */
