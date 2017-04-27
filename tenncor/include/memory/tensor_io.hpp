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

#include <string>
#include <iostream>
#include <fstream>

#include "graph/leaf/variable.hpp"

#ifndef TENNCOR_WRITER_HPP
#define TENNCOR_WRITER_HPP

namespace nnet {

//! write inodes should be ordered (bottom-up)
template<typename T>
bool write (std::vector<inode<T>*> serialvec, std::string fname = "default.pbx");

//! read inodes should be ordered (bottom-up)
//! returns false if fname is not a protobuf parsable file
//! nodes not found in serialized file will be found in deserialvec, otherwise found nodes are removed
template<typename T>
bool read (std::vector<inode<T>*>& deserialvec, std::string fname);

}

#include "../../src/memory/tensor_io.ipp"

#endif /* TENNCOR_WRITER_HPP */
