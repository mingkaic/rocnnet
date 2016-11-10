//
//  clip.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef clip_hpp
#define clip_hpp

#include "iunar_ops.hpp"

namespace nnet {

// CLIP

template <typename T>
class clip_by_norm : public iunar_elem_ops<T> {
	private:
		T cap;

	protected:
		// backward chaining for AD
		virtual void setup_gradient (void) {
			for (ccoms::subject* child : this->dependencies_) {
				if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
					this->grad = clip_by_norm<T>::make(arg->get_gradient(), cap);
				}
			}
		}

		clip_by_norm (ivariable<T>& var, std::string name) { this->copy(var, name); }
		std::string get_symb (void) { return "clip_norm"; }

		std::function<T(T)> get_op (void);
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		clip_by_norm (ivariable<T>* var, T cap) :
			iunar_ops(var), cap(cap) {}

		void set_bounds (T min, T max) { this->min = min; this->max = max; }

        clip_by_norm<T>* clone (std::string name = "") {
			return static_cast<clip_by_norm<T>*>(clone_impl(name));
		}
};

}

#include "../../../../src/graph/operation/unary/clip.ipp"

#endif /* clip_hpp */
