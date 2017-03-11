/*!
 *
 *  matmul.hpp
 *  cnnet
 *
 *  Purpose:
 *  the immutable matrix multiplication
 *  adds a jacobian transfer function
 *
 *  Created by Mingkai Chen on 2016-10-08
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/operation/immutable/elementary.hpp"
#include "graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_MATMUL_HPP
#define TENNCOR_MATMUL_HPP

namespace nnet
{

//! restricted to 2-d matrices with proper shapes
//! dimension 1 denote number of columns,
//! dimension 2 denote number of rows
template <typename T>
class matmul final : public operation<T>
{
public:
	//! override new matmul constructor
	//! in record to determine heap information
	static void* operator new (size_t size,
		inode<T>* a, inode<T>* b,
		bool transposeA = false,
		bool transposeB = false);

	// >>>> CONSTRUCTORS <<<<
	//! matrix multiplication constructor
	matmul (inode<T>* a, inode<T>* b,
		bool transposeA = false,
		bool transposeB = false);

	// >>>> CLONE, COPY && MOVE <<<<
	//! Clone function
	matmul<T>* clone (void);

	//! Declare move constructor to move over transfer functions
	matmul (matmul<T>&& other);

	//! Declare copy assignment to copy over transfer functions
	virtual matmul<T>& operator = (const matmul<T>& other);

	//! Declare move assignment to move over transfer functions
	virtual matmul<T>& operator = (matmul<T>&& other);

protected:
	// >>>> COPY CONSTRUCTOR <<<<
	//! Declare copy constructor to copy over transfer functions
	matmul (const matmul<T>& other);

	//! Implement clone function
	virtual inode<T>* clone_impl (void) const;

private:
	//! whether we transpose argument 1 before multiplication
	bool transposeA_;
	//! whether we transpose argument 2 before multiplication
	bool transposeB_;
};

}

#include "../../../../src/graph/operation/immutable/matmul.ipp"

#endif /* TENNCOR_MATMUL_HPP */
