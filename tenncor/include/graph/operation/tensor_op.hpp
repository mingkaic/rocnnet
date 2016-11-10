//
//  tensor_op.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-09.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifndef tensor_op_hpp
#define tensor_op_hpp

#include "graph/operation/ioperation.hpp"

namespace nnet {

// template <typename T>
// class collect_op : public ioperation<T> {
// 	protected:
// 		std::function<void(T&, T, size_t)> collect_;
// 		BUILD_DERIVE<T> der_;

// 		virtual void setup_gradient (void);
// 		virtual EVOKER_PTR<T> clone_impl (std::string name);

// 		virtual void shape_eval (void);

// 	public:
// 		tensor_op (std::vector<ivariable<T>*> args,
// 					std::function<void(T&, T)> op,
// 					BUILD_DERIVE<T> der,
// 					std::string name = "");

// 		// COPY
// 		std::shared_ptr<tensor_op<T> > clone (std::string name = "") {
// 			return std::static_pointer_cast<tensor_op<T>, ievoker<T> >(clone_impl(name));
// 		}
		
// 		// MOVES
// 		// TODO: implement

// 		virtual void update (void);
// };

}

#endif /* tensor_op_hpp */