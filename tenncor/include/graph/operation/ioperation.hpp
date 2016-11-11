//
//  operation.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <unordered_map>
#include <algorithm>
#include <random>
#include <functional>
#include <limits>
#include <memory>

#pragma once
#ifndef operation_hpp
#define operation_hpp

#include "../variable/variable.hpp"
#include "../observer/node.hpp"

namespace nnet {

template <typename T>
using BUILD_DERIVE = std::function<ivariable<T>*(std::vector<ivariable<T>*>)>;

// INTERFACE OPERATION

template <typename T>
class iunar_ops;
template <typename T>
class ibin_ops;
template <typename T>
class univar_func;

// inheritance join at ileaf_handler
template <typename T>
class ioperation : virtual public ivariable<T>, virtual public ccoms::iobserver {
	protected:
		ivariable<T>* grad_ = nullptr;

// DEPRECATE THESE TOO
		// TODO: find a way to move shape evaluation functions out of here
		// utility shape operations
		tensor_shape transpose_shape (const tensor_shape& ins) const;

		tensor_shape change_shape (
			const tensor_shape& ins,
			size_t index,
			double multiplier,
			size_t& below_dim,
			size_t& at_idx) const;

		tensor_shape get_matrix_shape (
			const tensor<T>& t1, const tensor<T>& t2,
			bool transposeA, bool transposeB,
			size_t& common_dim) const;

		// operator wrapper functions that handle variable scalar and shape
		// differences. returned tensor's ownership transfers to caller
		// reduce/filter functional
		template <typename U>
		void util_op (
			U& out, const tensor<T>& in,
			std::function<void(U&, T)> op) const;
		// collector functions while maintaining tensor shape
		void elem_op (
			tensor<T>& out,
			const tensor<T>& in,
			std::function<void(T&, T)> op) const;

// TODO DEPRECATE FROM HERE ==>
		tensor<T>* get_trace (const ivariable<T>& in) const;
		// 1 to 1 map functional
		tensor<T>* util_op (
			const tensor<T>& in,
			std::function<T(T)> op) const;
		// operator wrapper functions that restricts shapes to matrices and
		// retrieve raw value
		tensor<T>* transpose_op (
		    tensor<T> const & in) const;
		tensor<T>* extend_op (const tensor<T>& in, size_t index, size_t multiplier) const;
		tensor<T>* compress_op (
			const tensor<T>& in,
			signed index,
			std::function<T(const std::vector<T>&)> collector) const;
		tensor<T>* matmul_op (
			const tensor<T>& a,
			const tensor<T>& b,
		    bool transposeA,
		    bool transposeB) const;
// TO HERE <==

		// implement unique method of consuming input variables
		// to extract shape info
		virtual void shape_eval (void) = 0;
		
		// CONSTRUCTS THE GRADIENT TREE AND STORE ROOT IN MEMBER GRAD
		virtual void setup_gradient (void) = 0;

		virtual void copy (const ioperation<T>& other,
				   std::string name = "") {
			// if grad is not being observed, then and only then delete
			if (nullptr != grad_ && grad_->no_audience()) {
				delete grad_;
			}
			// shallow copy
			grad_ = other.grad_;
			ivariable<T>::copy(other, name);
		}
		
		virtual ievoker<T>* clone_impl (std::string name) = 0;

		friend class ivariable<T>;
		friend class placeholder<T>;
		friend class univar_func<T>;

	public:
		ioperation (std::vector<ivariable<T>*> dependencies, std::string name) :
				ivariable<T>(std::vector<size_t>{}, name), 
				iobserver(std::vector<ccoms::subject*>(dependencies_.begin(), dependencies_.end())) {
			this->short_circuit_ = false;
			if (session::pre_shape_eval()) {
				shape_eval();
			}
		}
		virtual ~ioperation (void) {
			if (nullptr != grad_) {
				delete grad_;
			}
		}

		// COPY
        ioperation<T>* clone (std::string name = "") {
			return static_cast<ioperation<T>*>(clone_impl(name));
		}
		virtual ioperation<T>& operator = (const ioperation<T>& other);
		
		virtual const tensor<T>& get_eval (void) {
			if (this->short_circuit_) {
				return this->ones;
			}
			return this->out_;
		}

		virtual ivariable<T>* get_gradient (void) {
			if (nullptr == grad_) {
				setup_gradient();
			}
			return grad_;
		}
};

}

#include "../../../src/graph/operation/ioperation.ipp"

#endif /* operation_hpp */
