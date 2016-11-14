//
//  tensor_jacobi.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef tensor_jacobi_hpp
#define tensor_jacobi_hpp

#include "tensor.hpp"

namespace nnet {

template <typename T>
class tensor_jacobi : public tensor<T> {
	private:
		bool transposeA_;
		bool transposeB_;
		ivariable<T>* k_ = nullptr; // no ownership on k
		std::vector<std::shared_ptr<ivariable<T> > > owner_;
		ivariable<T>* root_ = nullptr; // deleted when clear

		tensor_jacobi (const tensor_jacobi<T>& other) {
			this->copy(other);
		}

		void clear_ownership (void) {
			owner_.clear();
			root_ = nullptr;
		}

	protected:
		void copy (const tensor_jacobi<T>& other) {
			transposeA_ = other.transposeA_;
			transposeB_ = other.transposeB_;
			k_ = other.k_;
			owner_ = other.owner_;
			root_ = other.root_;
			tensor<T>::copy(other);
		}

		virtual tensor<T>* clone_impl (void) { return new tensor_jacobi<T>(*this); }

		virtual T* get_raw (void) {
			tensor<T>* src = root_->get_eval();
			this->copy(*src);
			return tensor<T>::get_raw();
		}

	public:
		tensor_jacobi (bool transposeA, bool transposeB) :
			transposeA_(transposeA), transposeB_(transposeB) {}
		virtual ~tensor_jacobi (void) { clear_ownership(); }

		tensor_jacobi<T>* clone (void) { return static_cast<tensor_jacobi<T>*>(clone_impl()); }

		void set_root (ivariable<T>* root) { k_ = root; }

		virtual const tensor_jacobi<T>& operator () (ivariable<T>* arga, ivariable<T>* argb) {
			// J(a, b) = d(a) * matmul(k, b^T) + d(b) * matmul(a^T, k)
			clear_ownership();
			// we're at evaluation time, so we can assess which derivative to take
			const tensor<T>* ag = arga->get_gradient()->get_eval();
			const tensor<T>* bg = argb->get_gradient()->get_eval();
			ivariable<T>* a = nullptr;
			if (ag) {
				a = new matmul<T>(k_, argb, transposeA_, !transposeB_);
				owner_.push_back(std::shared_ptr<ivariable<T> >(a));
				a = a * arga->get_gradient();
			}
			ivariable<T>* b = nullptr;
			if (bg) {
				b = new matmul<T>(arga, k_, !transposeA_, transposeB_);
				owner_.push_back(std::shared_ptr<ivariable<T> >(b));
				b = b * argb->get_gradient();
			}
			root_ = a + b;
			return *this;
		}
};

}

#endif /* tensor_jacobi_hpp */
