/*!
 *
 *  tensor_writer.hpp
 *  cnnet
 *
 *  Purpose:
 *  serialize tensor
 *
 *  Created by Mingkai Chen on 2017-04-25.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/leaf/variable.hpp"

#ifndef TENNCOR_WRITER_HPP
#define TENNCOR_WRITER_HPP

#include <string>
#include <iostream>
#include <fstream>

namespace nnet {

//! write inodes in order of serialvec by referencing protobuf objects labeled with <label + ":" + serialvec index>
template<typename T>
bool write_inorder (std::vector<inode<T>*>& serialvec, std::string label, std::string fname = "default.pbx");

//! read inodes in order of deserialvec by referencing protobuf objects labeled with <label + ":" + deserialvec index>
//! returns false if fname is not a protobuf parsable file
//! nodes not found in serialized file will be found in deserialvec, otherwise found nodes are removed
template<typename T>
bool read_inorder (std::vector<inode<T>*>& deserialvec, std::string label, std::string fname);

}

#include "../../src/memory/tensor_io.ipp"

#endif /* TENNCOR_WRITER_HPP */
