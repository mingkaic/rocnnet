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
			for (subject* child : this->dependencies_) {
				if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
					this->grad = clip_by_norm<T>::make(arg->get_gradient(), cap);
				}
			}
		}

		clip_by_norm (ivariable<T>& var, std::string name) { this->copy(var, name); }
		clip_by_norm (VAR_PTR<T> var, T cap) : cap(cap) { this->init(var); }

		std::string get_symb (void) { return "clip_norm"; }
		std::function<T(T)> get_op (void);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> var, T cap) {
			VAR_PTR<T> o = ivariable<T>::make_shared(new clip_by_norm(var, cap));
			VAR_PTR<T>* root = &o;
			// TODO: come up with a dryer solution to handling inherited attribute nodes (perhaps treat every node as inherited?)
			// have each argument evaluate interaction root
			ivariable<T>::set_interaction(var, root);
			return *root;
		}
		virtual ~clip_by_norm (void) {}

		void set_bounds (T min, T max) { this->min = min; this->max = max; }

		std::shared_ptr<clip_by_norm<T> > clone (std::string name = "") {
			return std::static_pointer_cast<clip_by_norm<T>, ievoker<T> >(clone_impl(name));
		}
};

}

#include "../../../../src/graph/operation/unary/clip.ipp"

#endif /* clip_hpp */
