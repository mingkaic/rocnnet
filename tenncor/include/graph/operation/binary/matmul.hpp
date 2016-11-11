//
//  mat_ops.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef matop_hpp
#define matop_hpp

#include "graph/operation/ioperation.hpp"
#include "../../inherited/inherited.hpp"

namespace nnet {

// MATRIX MULTIPLICATION

// restricted to 2-d matrices with proper shapes
// dimension 1 denote number of columns,
// dimension 2 denote number of rows
template <typename T>
class matmul : public ioperation<T> {
	private:
		bool transposeA_;
		bool transposeB_;

		matmul (const matmul<T>& other, std::string name);

	protected:
		// backward chaining for AD
		virtual void setup_gradient (void);
		virtual void shape_eval (void);
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		matmul (ivariable<T>* a, ivariable<T>* b, bool transposeA, bool transposeB);

		// COPY
        matmul<T>* clone (std::string name = "") {
			return static_cast<matmul<T>*>(clone_impl(name));
		}
		virtual matmul<T>& operator = (const ivariable<T>& other);
		
		// MOVES
		// TODO: implement

		virtual void update (void);
};

}

#include "../../../../src/graph/operation/binary/matmul.ipp"

#endif /* matop_hpp */
