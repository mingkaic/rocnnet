//
//  jacobian.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-06.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/operation/ioperation.hpp"
#include "tensor/tensor_jacobi.hpp"

#pragma once
#ifndef jacobian_hpp
#define jacobian_hpp

namespace nnet
{

// higher-dimensional unaligned operational tensor

// jacobian is really a tensor of tensors
// (each tensor being an abstraction of either raw data, or some system of equations)
// currently, tensor of tensor operation isn't well defined, so instead only accept jacobian vectors,
// vectors of tensors gradients (more specifically, gradients of matrix multiplication)

// the jacobian is a two part node, the publicly exposed version which always returns one as the evaluation
// and the private version which returns the jacobian tensor (tensor_jacobi)
// the jacobian tensor must be manually connected to other nodes via its set_root method

template <typename T>
class jacobian : public ioperation<T>
{
	private:
		class hidden_jacobi;

		std::unique_ptr<hidden_jacobi> hidden_;

	protected:
		// jacobian is inherently a gradient agent (it exists in some first order or above derivative function)
		// TODO will do nothing until second/nth order derivative is implemented
		virtual void setup_gradient (void) {}
		// never used
		virtual tensorshape shape_eval (void) { return std::vector<size_t>{}; }

		jacobian (const jacobian<T>& other, std::string name);
		virtual ivariable<T>* clone_impl (std::string name);

		virtual bool channel (std::stack<ivariable<T>*>& jacobi);

		// protect jacobian constructor to ensure heap allocation
		jacobian (ivariable<T>* arga, ivariable<T>* argb,
			bool transposeA, bool transposeB, std::string name);

	public:
		static ivariable<T>* build (ivariable<T>* a, ivariable<T>* b,
			bool transposeA, bool transposeB, std::string name = "")
		{
			return new jacobian<T>(a, b, transposeA, transposeB, name);
		}

		// COPY
		jacobian<T>* clone (std::string name = "");
		virtual jacobian<T>& operator = (const jacobian<T>& other);

		// MOVE

		virtual void update (ccoms::subject* caller);
};

}

#include "../../../../src/graph/operation/special/jacobian.ipp"

#endif /* jacobian_hpp */
