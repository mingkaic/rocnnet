/*!
 *
 *  const_immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  an immutable extension that
 *  clears all leaves and mimic constant behavior
 *
 *  Created by Mingkai Chen on 2017-09-17.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef CONST_IMMUTABLE_HPP
#define CONST_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
class const_immutable : public immutable<T>
{
public:
	static const_immutable<T>* get (inode<T>* x);

private:
	const_immutable (inode<T>* x);
};

}

#include "../../../../src/graph/connector/immutable/const_immutable.ipp"

#endif /* CONST_IMMUTABLE_HPP */
