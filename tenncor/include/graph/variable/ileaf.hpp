//
//  ileaf.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef ileaf_hpp
#define ileaf_hpp

#include "ivariable.hpp"

namespace nnet {

// INITIALIZER MANAGING INTERFACE
// Leaf Nodes

template <typename T>
class ileaf : public ivariable<T> {
	protected:
		initializer<T>* init_ = nullptr;
		bool is_init = false;
		
		// GRADIENT END HANDLING
		// DEPRECATED
		// WEAK_VAR_PTR<T> grad;
		// virtual void make_gradient (VAR_PTR<T>& safety_ref) {
		// 	safety_ref = constant<T>::make(0);
		// 	this->set_gradient(safety_ref);
		// }

		// virtual void set_gradient (VAR_PTR<T> g) {
		// 	if (grad.expired() && nullptr != g) {
		// 		grad = g;
		// 		ivariable<T>::set_gradient(g);
		// 	}
		// }

		// used by assignment operators to freely initialized inner tensor
		struct open_init;

		virtual void copy (const ileaf<T>& other,
				   std::string name = "") {
			init_ = other.init_->clone();
			init_ = other.is_init;
			ivariable<T>::copy(other, name);
		}

	public:
		ileaf (const tensor_shape& shape, initializer<T>* init, std::string name) : 
			ivariable(shape, name), init_(init) {}
		virtual ~ileaf (void) {
			if (nullptr != init_) {
				delete init_;
			}
		}

		// COPY
		virtual ileaf<T>& operator = (const VAR_PTR<T>& other);
		std::shared_ptr<ileaf<T> > clone (std::string name = "") {
			return std::static_pointer_cast<ileaf<T>, ievoker<T> >(this->clone_impl(name));
		}

		// MOVES
		// todo: implement move clone
		virtual placeholder<T>& operator = (placeholder<T>&& other) = default;

		// GET INFO
		bool can_init (void) const { return init_ != nullptr; }
		
		// DATA EXPOSURE TO PARENT/DEPENDENT NODES

		virtual ivariable<T>* get_gradient (void) { return this; }
		
		// DEPRECATED
// 		virtual const tensor<T>& eval (void) {
// 			assert(is_init);
// 			return this->out_;
// 		}
};

}

#include "../../../src/graph/variable/ileaf.ipp"

#endif /* ileaf_hpp */
