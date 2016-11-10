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
		ivariable<T>* a = nullptr;
		ivariable<T>* b = nullptr;
		bool transposeA;
		bool transposeB;

	protected:
		// backward chaining for AD
		virtual void setup_gradient (void);

		virtual void shape_eval (void);
		matmul (const matmul<T>& other, std::string name);
		matmul (ivariable<T>* a, ivariable<T>* b, bool transposeA, bool transposeB);

		virtual ievoker<T>* clone_impl (std::string name);

	public:
		static ivariable<T>* make (ivariable<T>* a, ivariable<T>* b, bool transposeA = false, bool transposeB = false) {
			ivariable<T>* o = ivariable<T>::make_shared(new matmul(a, b, transposeA, transposeB));
			ivariable<T>* root = &o;
			// TODO: come up with a dryer solution to handling inherited attribute nodes (perhaps treat every node as inherited?)
			// have each argument evaluate interaction root
			// multiple inherited attributes is currently undefined behavior... (probably going to return bad tensor)
			ivariable<T>::set_interaction(a, root);
			ivariable<T>::set_interaction(b, root);
			return *root;
		}

		virtual matmul<T>& operator = (const ivariable<T>& other);

        matmul<T>* clone (std::string name = "") {
			return static_cast<matmul<T>*>(clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

}

#include "../../../../src/graph/operation/binary/matmul.ipp"

#endif /* matop_hpp */
