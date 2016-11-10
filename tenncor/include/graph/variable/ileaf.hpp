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

#include "graph/ivariable.hpp"

namespace nnet {

// INITIALIZER MANAGING INTERFACE
// Leaf Nodes

template <typename T>
class ileaf : public ivariable<T> {
	protected:
		initializer<T>* init_ = nullptr;
		bool is_init = false;

		// used by assignment operators to freely initialized inner tensor
		struct open_init;

		virtual void copy (const ileaf<T>& other,
				   std::string name = "") {
            if (nullptr != init_) {
                delete init_;
            }
			init_ = other.init_->clone();
			init_ = other.is_init;
			ivariable<T>::copy(other, name);
		}

	public:
		ileaf (const tensor_shape& shape, initializer<T>* init, std::string name) : 
			ivariable<T>(shape, name), init_(init) {}
		virtual ~ileaf (void) {
			if (nullptr != init_) {
				delete init_;
			}
		}

		// MOVES
		// todo: implement move clone

		// GET INFO
		bool can_init (void) const { return init_ != nullptr; }

		// DATA EXPOSURE TO PARENT/DEPENDENT NODES

		virtual ivariable<T>* get_gradient (void) {
			return this;
		}
};

}

#include "../../../src/graph/variable/ileaf.ipp"

#endif /* ileaf_hpp */
