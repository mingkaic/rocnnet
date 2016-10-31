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

namespace nnet {

// MATRIX MULTIPLICATION

// restricted to 2-d matrices with proper shapes
// dimension 1 denote number of columns,
// dimension 2 denote number of rows
template <typename T>
class matmul : public ioperation<T> {
	private:
		VAR_PTR<T> a = nullptr;
		VAR_PTR<T> b = nullptr;
		bool transposeA;
		bool transposeB;

	protected:
		// backward chaining for AD
		virtual void make_gradient (VAR_PTR<T>& safety_ref);
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood) {
			if (a.get() == food) a = newfood;
			if (b.get() == food) b = newfood;
		}

		virtual void shape_eval (void);
		matmul (const matmul<T>& other, std::string name);
		matmul (VAR_PTR<T> a, VAR_PTR<T> b, bool transposeA, bool transposeB);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> a, VAR_PTR<T> b, bool transposeA = false, bool transposeB = false) {
			return ivariable<T>::make_shared(new matmul(a, b, transposeA, transposeB));
		}

		virtual matmul<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<matmul<T> > clone (std::string name = "") {
			return std::static_pointer_cast<matmul<T>, ievoker<T> >(clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

}

#include "../../../../src/graph/operation/binary/matmul.ipp"

#endif /* matop_hpp */