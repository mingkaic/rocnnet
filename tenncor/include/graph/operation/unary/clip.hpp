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
class clip_by_norm : public iunar_ops<T> {
	private:
		T cap;

		clip_by_norm (clip_by_norm<T>& other, std::string name) {
			cap = other.cap;
			ioperation<T>::copy(other, name);
		}

	protected:
		// backward chaining for AD
		virtual void setup_gradient (void) {
			for (ccoms::subject* child : this->dependencies_) {
				if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
					this->grad_ = new clip_by_norm<T>(arg->get_gradient(), cap);
				}
			}
		}
		std::string get_symb (void) { return "clip_norm"; }
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		clip_by_norm (ivariable<T>* var, T cap) :
			iunar_ops(var), cap(cap) {}

        clip_by_norm<T>* clone (std::string name = "") {
			return static_cast<clip_by_norm<T>*>(clone_impl(name));
		}
		clip_by_norm& operator = (const clip_by_norm& other) {
			if (this != &other) {
				cap = other.cap;
				ioperation<T>::copy(other);
			}
			return *this;
		}

		void set_bounds (T min, T max) { this->min = min; this->max = max; }
		
		virtual void update (void);
};

}

#include "../../../../src/graph/operation/unary/clip.ipp"

#endif /* clip_hpp */
